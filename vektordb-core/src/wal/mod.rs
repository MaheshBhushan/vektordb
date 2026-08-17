//! Write-ahead log.
//!
//! Record framing:
//!   [len: u32][crc32: u32][payload: len bytes]
//!   payload = [lsn: u64][op: u8][op body]
//! CRC covers the payload. Recovery walks records until EOF, a short read,
//! or a CRC mismatch; everything from the first bad record on is a torn
//! tail from a crash mid-write and is truncated — that is the expected
//! contract, not an error: a record is acked only after fdatasync, so a
//! torn record was never acked.
//!
//! Ops: only `Insert` in v1 (HNSW deletion is future work), plus the
//! checkpoint watermark living in the snapshot file, not the log.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::Result;

const OP_INSERT: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    /// fdatasync after every record: an acked insert survives any crash.
    Always,
    /// No explicit sync (bulk loads where the caller checkpoints at the end).
    Never,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WalOp {
    Insert { id: u64, vector: Vec<f32> },
}

pub struct Wal {
    file: File,
    policy: SyncPolicy,
    next_lsn: u64,
}

impl Wal {
    /// Open (or create) the log, scan it for intact records, truncate any
    /// torn tail, and return the WAL positioned for appending together with
    /// the surviving records.
    pub fn open<P: AsRef<Path>>(path: P, policy: SyncPolicy) -> Result<(Self, Vec<(u64, WalOp)>)> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;

        let mut records = Vec::new();
        let mut good_end: u64 = 0;
        let mut next_lsn: u64 = 0;
        {
            let len = file.metadata()?.len();
            let mut reader = BufReader::new(&mut file);
            loop {
                match read_record(&mut reader, len, good_end) {
                    Some((consumed, lsn, op)) => {
                        good_end += consumed;
                        next_lsn = next_lsn.max(lsn + 1);
                        records.push((lsn, op));
                    }
                    None => break,
                }
            }
        }
        if file.metadata()?.len() > good_end {
            file.set_len(good_end)?; // drop the torn tail
            file.sync_data()?;
        }
        file.seek(SeekFrom::Start(good_end))?;
        Ok((
            Self {
                file,
                policy,
                next_lsn,
            },
            records,
        ))
    }

    pub fn next_lsn(&self) -> u64 {
        self.next_lsn
    }

    /// Floor the LSN counter. Called with the snapshot watermark on open:
    /// after a checkpoint resets the log, the counter survives only in
    /// memory, so a crash-then-reopen would otherwise restart LSNs at 0 —
    /// below the watermark — and the *next* recovery would skip those
    /// records as "already covered by the snapshot", silently dropping
    /// acked inserts. (Found by the SIGKILL harness.)
    pub fn ensure_lsn_at_least(&mut self, min: u64) {
        self.next_lsn = self.next_lsn.max(min);
    }

    /// Append one record. Returns its LSN after the configured sync — once
    /// this returns under `SyncPolicy::Always`, the record is durable.
    pub fn append(&mut self, op: &WalOp) -> Result<u64> {
        let lsn = self.next_lsn;
        let mut payload = Vec::with_capacity(64);
        payload.extend_from_slice(&lsn.to_le_bytes());
        match op {
            WalOp::Insert { id, vector } => {
                payload.push(OP_INSERT);
                payload.extend_from_slice(&id.to_le_bytes());
                payload.extend_from_slice(&(vector.len() as u32).to_le_bytes());
                for x in vector {
                    payload.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        let crc = crc32fast::hash(&payload);
        let mut frame = Vec::with_capacity(8 + payload.len());
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(&crc.to_le_bytes());
        frame.extend_from_slice(&payload);
        self.file.write_all(&frame)?;
        if self.policy == SyncPolicy::Always {
            self.file.sync_data()?;
        }
        self.next_lsn += 1;
        Ok(lsn)
    }

    /// Discard all records (called after a checkpoint made them redundant).
    /// LSNs keep increasing across resets so the snapshot watermark stays
    /// unambiguous.
    pub fn reset(&mut self) -> Result<()> {
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.sync_data()?;
        Ok(())
    }

    pub fn sync(&mut self) -> Result<()> {
        self.file.sync_data()?;
        Ok(())
    }
}

/// Try to read one record at `pos`; `None` on EOF / torn tail.
fn read_record<R: Read>(reader: &mut R, file_len: u64, pos: u64) -> Option<(u64, u64, WalOp)> {
    if pos + 8 > file_len {
        return None;
    }
    let mut head = [0u8; 8];
    reader.read_exact(&mut head).ok()?;
    let len = u32::from_le_bytes(head[0..4].try_into().unwrap()) as u64;
    let crc = u32::from_le_bytes(head[4..8].try_into().unwrap());
    if len < 9 || pos + 8 + len > file_len {
        return None; // impossible length: torn or garbage
    }
    let mut payload = vec![0u8; len as usize];
    reader.read_exact(&mut payload).ok()?;
    if crc32fast::hash(&payload) != crc {
        return None;
    }
    let lsn = u64::from_le_bytes(payload[0..8].try_into().unwrap());
    let op = match payload[8] {
        OP_INSERT => {
            if payload.len() < 21 {
                return None;
            }
            let id = u64::from_le_bytes(payload[9..17].try_into().unwrap());
            let dim = u32::from_le_bytes(payload[17..21].try_into().unwrap()) as usize;
            if payload.len() != 21 + dim * 4 {
                return None;
            }
            let vector = payload[21..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            WalOp::Insert { id, vector }
        }
        _ => return None,
    };
    Some((8 + len, lsn, op))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(id: u64) -> WalOp {
        WalOp::Insert {
            id,
            vector: vec![id as f32; 8],
        }
    }

    #[test]
    fn append_reopen_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        {
            let (mut wal, recs) = Wal::open(&path, SyncPolicy::Always).unwrap();
            assert!(recs.is_empty());
            for i in 0..50 {
                assert_eq!(wal.append(&op(i)).unwrap(), i);
            }
        }
        let (wal, recs) = Wal::open(&path, SyncPolicy::Always).unwrap();
        assert_eq!(recs.len(), 50);
        assert_eq!(wal.next_lsn(), 50);
        for (i, (lsn, o)) in recs.iter().enumerate() {
            assert_eq!(*lsn, i as u64);
            assert_eq!(*o, op(i as u64));
        }
    }

    #[test]
    fn torn_tail_is_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        {
            let (mut wal, _) = Wal::open(&path, SyncPolicy::Never).unwrap();
            for i in 0..10 {
                wal.append(&op(i)).unwrap();
            }
            wal.sync().unwrap();
        }
        // Chop bytes off the last record to simulate a crash mid-write.
        let full = std::fs::metadata(&path).unwrap().len();
        let f = OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(full - 7).unwrap();
        drop(f);

        let (mut wal, recs) = Wal::open(&path, SyncPolicy::Always).unwrap();
        assert_eq!(recs.len(), 9, "torn record must be dropped");
        // And the log keeps working after truncation.
        assert_eq!(wal.next_lsn(), 9);
        wal.append(&op(99)).unwrap();
        drop(wal);
        let (_, recs) = Wal::open(&path, SyncPolicy::Always).unwrap();
        assert_eq!(recs.len(), 10);
    }

    #[test]
    fn corrupted_middle_stops_replay_there() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        {
            let (mut wal, _) = Wal::open(&path, SyncPolicy::Never).unwrap();
            for i in 0..10 {
                wal.append(&op(i)).unwrap();
            }
            wal.sync().unwrap();
        }
        // Flip a byte in record 5's payload.
        let mut bytes = std::fs::read(&path).unwrap();
        let rec_size = bytes.len() / 10;
        bytes[5 * rec_size + 12] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        let (_, recs) = Wal::open(&path, SyncPolicy::Always).unwrap();
        assert_eq!(recs.len(), 5, "replay must stop at the corrupt record");
    }

    #[test]
    fn reset_clears_but_lsn_advances() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let (mut wal, _) = Wal::open(&path, SyncPolicy::Always).unwrap();
        for i in 0..5 {
            wal.append(&op(i)).unwrap();
        }
        wal.reset().unwrap();
        assert_eq!(
            wal.append(&op(100)).unwrap(),
            5,
            "LSNs continue after reset"
        );
        drop(wal);
        let (_, recs) = Wal::open(&path, SyncPolicy::Always).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].0, 5);
    }
}
