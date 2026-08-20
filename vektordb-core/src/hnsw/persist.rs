//! HNSW snapshot serialization (checkpoints).
//!
//! Format (little-endian throughout):
//!   magic u32 | version u32 | m u32 | ef_construction u32 | metric u8 |
//!   wal_lsn u64 | count u64 | entry u64 (packed) |
//!   per node: level u32, then per layer: len u32 + ids u64...
//!   crc32 u32 over everything before it
//!
//! Written to a temp file, fsynced, then renamed over the target — a crash
//! mid-checkpoint leaves the previous snapshot intact.
//!
//! `save` must not run concurrently with inserts (the `Db` layer enforces
//! this with its maintenance lock); loads happen before the index is shared.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::Ordering;

use crossbeam_epoch::{self as epoch, Owned};

use crate::distance::Metric;
use crate::error::{Error, Result};

use super::{pack_entry, unpack_entry, Hnsw, HnswConfig, Node, ENTRY_NONE};

const MAGIC: u32 = 0x564B_534E; // "VKSN"
const VERSION: u32 = 1;

fn verify_crc(file: &mut File) -> Result<u64> {
    let file_len = file.metadata()?.len();
    if file_len < 4 {
        return Err(Error::Corrupt("snapshot: truncated".into()));
    }

    let payload_len = file_len - 4;
    let mut reader = BufReader::new(&mut *file);
    let mut hasher = crc32fast::Hasher::new();
    let mut remaining = payload_len;
    let mut buf = [0u8; 64 * 1024];
    while remaining > 0 {
        let take = remaining.min(buf.len() as u64) as usize;
        reader.read_exact(&mut buf[..take])?;
        hasher.update(&buf[..take]);
        remaining -= take as u64;
    }
    let mut crc = [0u8; 4];
    reader.read_exact(&mut crc)?;
    if u32::from_le_bytes(crc) != hasher.finalize() {
        return Err(Error::Corrupt("snapshot: checksum mismatch".into()));
    }
    drop(reader);
    file.seek(SeekFrom::Start(0))?;
    Ok(file_len)
}

fn metric_tag(m: Metric) -> u8 {
    match m {
        Metric::L2 => 0,
        Metric::Dot => 1,
        Metric::Cosine => 2,
    }
}

fn metric_from(tag: u8) -> Result<Metric> {
    Ok(match tag {
        0 => Metric::L2,
        1 => Metric::Dot,
        2 => Metric::Cosine,
        _ => return Err(Error::Corrupt("unknown metric tag".into())),
    })
}

struct CrcWriter<W: Write> {
    inner: W,
    hasher: crc32fast::Hasher,
}

impl<W: Write> CrcWriter<W> {
    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        self.hasher.update(bytes);
        self.inner.write_all(bytes)?;
        Ok(())
    }
}

struct CrcReader<R: Read> {
    inner: R,
    hasher: crc32fast::Hasher,
}

impl<R: Read> CrcReader<R> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut buf = [0u8; N];
        self.inner.read_exact(&mut buf)?;
        self.hasher.update(&buf);
        Ok(buf)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take::<4>()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take::<8>()?))
    }
}

impl Hnsw {
    /// Snapshot the graph. `wal_lsn` is the first LSN *not* covered by this
    /// snapshot (replay resumes there).
    pub fn save<P: AsRef<Path>>(&self, path: P, wal_lsn: u64) -> Result<()> {
        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        let mut w = CrcWriter {
            inner: BufWriter::new(file),
            hasher: crc32fast::Hasher::new(),
        };

        w.put(&MAGIC.to_le_bytes())?;
        w.put(&VERSION.to_le_bytes())?;
        w.put(&(self.config.m as u32).to_le_bytes())?;
        w.put(&(self.config.ef_construction as u32).to_le_bytes())?;
        w.put(&[metric_tag(self.config.metric)])?;
        w.put(&wal_lsn.to_le_bytes())?;

        let count = self.nodes.len() as u64;
        w.put(&count.to_le_bytes())?;
        w.put(&self.entry.load(Ordering::Acquire).to_le_bytes())?;

        let guard = epoch::pin();
        for i in 0..count {
            let node = self.node(i);
            w.put(&(node.level() as u32).to_le_bytes())?;
            for layer in 0..=node.level() {
                let p = node.links[layer].load(Ordering::Acquire, &guard);
                let links: &[u64] = if p.is_null() {
                    &[]
                } else {
                    unsafe { p.deref() }
                };
                w.put(&(links.len() as u32).to_le_bytes())?;
                for &nb in links {
                    w.put(&nb.to_le_bytes())?;
                }
            }
        }

        let crc = w.hasher.clone().finalize();
        w.inner.write_all(&crc.to_le_bytes())?;
        let file = w
            .inner
            .into_inner()
            .map_err(|e| Error::Io(e.into_error()))?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, path)?;
        // fsync the directory so the rename itself is durable.
        if let Some(dir) = path.parent() {
            File::open(dir)?.sync_all()?;
        }
        Ok(())
    }

    /// Load a snapshot. Returns the rebuilt index and the WAL LSN replay
    /// should resume from.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<(Hnsw, u64)> {
        let path = path.as_ref();
        // Verify the complete file before trusting lengths or allocating.
        let mut file = File::open(path)?;
        let file_len = verify_crc(&mut file)?;
        let mut r = CrcReader {
            inner: BufReader::new(file),
            hasher: crc32fast::Hasher::new(),
        };

        if r.u32()? != MAGIC {
            return Err(Error::Corrupt("snapshot: bad magic".into()));
        }
        if r.u32()? != VERSION {
            return Err(Error::Corrupt("snapshot: unsupported version".into()));
        }
        let m = r.u32()? as usize;
        let ef_construction = r.u32()? as usize;
        let metric = metric_from(r.take::<1>()?[0])?;
        let wal_lsn = r.u64()?;
        let count = r.u64()?;
        let entry = r.u64()?;

        if !(2..=4096).contains(&m) {
            return Err(Error::Corrupt("snapshot: HNSW M out of range".into()));
        }
        if count > file_len / 8 {
            return Err(Error::Corrupt("snapshot: impossible node count".into()));
        }

        let index = Hnsw::new(HnswConfig {
            m,
            ef_construction,
            metric,
        })?;
        let guard = epoch::pin();
        for i in 0..count {
            let level = r.u32()? as usize;
            if level > 63 {
                return Err(Error::Corrupt("snapshot: absurd level".into()));
            }
            let node = Node::new(level);
            for layer in 0..=level {
                let len = r.u32()? as usize;
                let max_degree = if layer == 0 { 2 * m } else { m };
                if len > max_degree {
                    return Err(Error::Corrupt("snapshot: absurd degree".into()));
                }
                let mut links = Vec::with_capacity(len);
                for _ in 0..len {
                    let nb = r.u64()?;
                    if nb >= count {
                        return Err(Error::Corrupt("snapshot: link out of range".into()));
                    }
                    links.push(nb);
                }
                if !links.is_empty() {
                    node.links[layer].store(Owned::new(links), Ordering::Relaxed);
                }
            }
            let _ = &guard;
            index.nodes.set(i as usize, node);
        }

        let expected = r.hasher.clone().finalize();
        let mut crc_buf = [0u8; 4];
        r.inner.read_exact(&mut crc_buf)?;
        if u32::from_le_bytes(crc_buf) != expected {
            return Err(Error::Corrupt("snapshot: checksum mismatch".into()));
        }
        let mut trailing = [0u8; 1];
        if r.inner.read(&mut trailing)? != 0 {
            return Err(Error::Corrupt("snapshot: trailing bytes".into()));
        }

        if entry != ENTRY_NONE {
            let (id, level) = unpack_entry(entry);
            if id >= count {
                return Err(Error::Corrupt("snapshot: entry out of range".into()));
            }
            if level != index.node(id).level() {
                return Err(Error::Corrupt("snapshot: entry level mismatch".into()));
            }
            index.entry.store(pack_entry(id, level), Ordering::Release);
        }
        Ok((index, wal_lsn))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::VectorStore;
    use rand::SeedableRng;

    #[test]
    fn save_load_round_trip_preserves_search() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::create(dir.path().join("v.store"), 12).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let index = Hnsw::new(HnswConfig {
            m: 8,
            ef_construction: 64,
            metric: Metric::L2,
        })
        .unwrap();
        for i in 0..2000 {
            let v: Vec<f32> = (0..12).map(|j| ((i * 12 + j) % 97) as f32 * 0.1).collect();
            let id = store.append(&v).unwrap();
            index.insert(&store, id, &mut rng);
        }
        let snap = dir.path().join("index.snap");
        index.save(&snap, 12345).unwrap();

        let (loaded, lsn) = Hnsw::load(&snap).unwrap();
        assert_eq!(lsn, 12345);
        assert_eq!(loaded.len(), 2000);
        // Same graph => identical search results.
        for q in 0..20 {
            let query: Vec<f32> = (0..12).map(|j| ((q * 7 + j) % 31) as f32 * 0.3).collect();
            let a = index.search(&store, &query, 10, 64);
            let b = loaded.search(&store, &query, 10, 64);
            assert_eq!(a, b, "query {q} diverged after reload");
        }
    }

    #[test]
    fn corrupted_snapshot_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::create(dir.path().join("v.store"), 4).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let index = Hnsw::new(HnswConfig::default()).unwrap();
        for i in 0..100 {
            let id = store.append(&[i as f32; 4]).unwrap();
            index.insert(&store, id, &mut rng);
        }
        let snap = dir.path().join("index.snap");
        index.save(&snap, 0).unwrap();

        let mut bytes = std::fs::read(&snap).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&snap, &bytes).unwrap();
        assert!(
            Hnsw::load(&snap).is_err(),
            "bit flip must not load silently"
        );
    }

    #[test]
    fn checksum_valid_snapshot_with_invalid_config_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let snap = dir.path().join("index.snap");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes()); // invalid M
        bytes.extend_from_slice(&64u32.to_le_bytes());
        bytes.push(metric_tag(Metric::L2));
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&ENTRY_NONE.to_le_bytes());
        bytes.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
        std::fs::write(&snap, bytes).unwrap();

        let message = match Hnsw::load(&snap) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("invalid snapshot config must be rejected"),
        };
        assert!(
            message.contains("M out of range"),
            "unhelpful error: {message}"
        );
    }
}
