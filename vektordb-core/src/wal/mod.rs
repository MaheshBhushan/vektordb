//! Write-ahead log.
//!
//! Record framing:
//!   [len: u32][crc32: u32][payload: len bytes]
//!   payload = [lsn: u64][op: u8][op body]
//! CRC covers the payload. Recovery walks records until EOF and then has to
//! decide, for anything that isn't a clean record, whether it is *torn* or
//! *corrupt* — because the two have opposite correct responses:
//!
//! - **Torn**: the bad frame is the last thing in the file. A crash mid-write
//!   can leave a partial header, a partial payload, or (if the page cache
//!   persisted sectors out of order) a full-length frame with garbage inside.
//!   None of it was ever acked, because an ack happens only after fdatasync.
//!   Truncating is correct and lossless.
//! - **Corrupt**: the bad frame has intact-looking data *after* it, so it was
//!   fully written and fsynced — it was acked — and something later damaged
//!   it. Truncating here would silently destroy every acked record that
//!   follows, permanently. So this is an error, and recovery refuses to open.
//!
//! Earlier versions treated both cases as a torn tail, which meant one
//! flipped bit mid-log silently dropped every record after it and then wrote
//! the truncation to disk.
//!
//! Ops: only `Insert` in v1 (HNSW deletion is future work), plus the
//! checkpoint watermark living in the snapshot file, not the log.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{Error, Result};

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
                    Step::Record { consumed, lsn, op } => {
                        good_end += consumed;
                        next_lsn = next_lsn.max(lsn + 1);
                        records.push((lsn, op));
                    }
                    Step::Stopped(Stop::Eof) | Step::Stopped(Stop::Torn) => break,
                    Step::Stopped(Stop::Corrupt(why)) => {
                        // Refusing to open is the whole point: the records
                        // after this one were acked, and truncating to
                        // `good_end` would destroy them irrecoverably.
                        return Err(Error::Corrupt(format!(
                            "wal: {why} at byte offset {good_end}, with {} intact \
                             record(s) before it and more data after it; this is \
                             damage to already-acked records, not a torn tail from \
                             a crash. Refusing to truncate.",
                            records.len()
                        )));
                    }
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

/// Why the recovery walk stopped at `pos`.
enum Stop {
    /// Clean end of the log.
    Eof,
    /// Incomplete or damaged frame that is the *last* thing in the file:
    /// a crash mid-write. Safe to truncate — it was never acked.
    Torn,
    /// Damaged frame with more data after it: it was acked, then rotted.
    /// Truncating would destroy acked records, so this fails recovery.
    Corrupt(&'static str),
}

enum Step {
    Record { consumed: u64, lsn: u64, op: WalOp },
    Stopped(Stop),
}

/// Read one record starting at `pos`, or explain why we can't.
fn read_record<R: Read>(reader: &mut R, file_len: u64, pos: u64) -> Step {
    use Step::{Record, Stopped};

    if pos == file_len {
        return Stopped(Stop::Eof);
    }
    if pos + 8 > file_len {
        return Stopped(Stop::Torn); // partial header
    }
    let mut head = [0u8; 8];
    if reader.read_exact(&mut head).is_err() {
        return Stopped(Stop::Torn);
    }
    let len = u32::from_le_bytes(head[0..4].try_into().unwrap()) as u64;
    let crc = u32::from_le_bytes(head[4..8].try_into().unwrap());

    // A length that can't be real means the header itself is damaged. We
    // can't trust it to tell us where this frame ends, so we can only ask
    // whether anything follows the header at all.
    if len < 9 {
        return Stopped(if pos + 8 == file_len {
            Stop::Torn
        } else {
            Stop::Corrupt("impossible record length")
        });
    }
    if pos + 8 + len > file_len {
        return Stopped(Stop::Torn); // payload truncated by the crash
    }

    // The frame is wholly inside the file. From here on, "is this the last
    // frame?" is what separates a torn tail from real corruption.
    let is_tail = pos + 8 + len == file_len;
    let damaged = |why: &'static str| {
        Stopped(if is_tail {
            Stop::Torn
        } else {
            Stop::Corrupt(why)
        })
    };

    let mut payload = vec![0u8; len as usize];
    if reader.read_exact(&mut payload).is_err() {
        return Stopped(Stop::Torn);
    }
    if crc32fast::hash(&payload) != crc {
        return damaged("checksum mismatch");
    }
    let lsn = u64::from_le_bytes(payload[0..8].try_into().unwrap());
    let op = match payload[8] {
        OP_INSERT => {
            if payload.len() < 21 {
                return damaged("insert record too short");
            }
            let id = u64::from_le_bytes(payload[9..17].try_into().unwrap());
            let dim = u32::from_le_bytes(payload[17..21].try_into().unwrap()) as usize;
            if payload.len() != 21 + dim * 4 {
                return damaged("insert record length disagrees with its dim");
            }
            let vector = payload[21..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            WalOp::Insert { id, vector }
        }
        _ => return damaged("unknown op tag"),
    };
    Record {
        consumed: 8 + len,
        lsn,
        op,
    }
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
    fn corrupted_middle_is_an_error_not_silent_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        {
            let (mut wal, _) = Wal::open(&path, SyncPolicy::Never).unwrap();
            for i in 0..10 {
                wal.append(&op(i)).unwrap();
            }
            wal.sync().unwrap();
        }
        let before = std::fs::read(&path).unwrap();
        // Flip a byte in record 5's payload. Records 6-9 sit after it, fully
        // written and fsynced, so they were acked.
        let mut bytes = before.clone();
        let rec_size = bytes.len() / 10;
        bytes[5 * rec_size + 12] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        let msg = match Wal::open(&path, SyncPolicy::Always) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("mid-log corruption must not be reported as success"),
        };
        assert!(msg.contains("checksum mismatch"), "unhelpful error: {msg}");
        assert!(
            msg.contains("Refusing to truncate"),
            "unhelpful error: {msg}"
        );

        // And the failed open must not have modified the file: the acked
        // records after the damage are still there to be recovered by hand.
        assert_eq!(
            std::fs::read(&path).unwrap().len(),
            bytes.len(),
            "a failed recovery must not truncate anything"
        );
    }

    #[test]
    fn corrupt_last_record_is_treated_as_a_torn_tail() {
        // A crash can leave the final frame full-length but with garbage in
        // it (sectors persisted out of order). Nothing follows it, so it was
        // never acked and truncating is correct -- this is the case that must
        // NOT become an error, or every real crash would refuse to reopen.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        {
            let (mut wal, _) = Wal::open(&path, SyncPolicy::Never).unwrap();
            for i in 0..10 {
                wal.append(&op(i)).unwrap();
            }
            wal.sync().unwrap();
        }
        let mut bytes = std::fs::read(&path).unwrap();
        let rec_size = bytes.len() / 10;
        let last = 9 * rec_size;
        bytes[last + 12] ^= 0xFF; // corrupt the tail frame's payload
        std::fs::write(&path, &bytes).unwrap();

        let (mut wal, recs) = Wal::open(&path, SyncPolicy::Always).unwrap();
        assert_eq!(recs.len(), 9, "torn tail dropped, earlier records kept");
        assert_eq!(wal.next_lsn(), 9);
        wal.append(&op(99)).unwrap(); // and the log still works
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
