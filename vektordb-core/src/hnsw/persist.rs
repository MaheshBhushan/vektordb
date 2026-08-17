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
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::Ordering;

use crossbeam_epoch::{self as epoch, Owned};

use crate::distance::Metric;
use crate::error::{Error, Result};

use super::{pack_entry, unpack_entry, Hnsw, HnswConfig, Node, ENTRY_NONE};

const MAGIC: u32 = 0x564B_534E; // "VKSN"
const VERSION: u32 = 1;

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
        let file = File::open(path)?;
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

        let index = Hnsw::new(HnswConfig {
            m,
            ef_construction,
            metric,
        });
        let guard = epoch::pin();
        for i in 0..count {
            let level = r.u32()? as usize;
            if level > 63 {
                return Err(Error::Corrupt("snapshot: absurd level".into()));
            }
            let node = Node::new(level);
            for layer in 0..=level {
                let len = r.u32()? as usize;
                if len > 4 * m.max(1) + 64 {
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

        if entry != ENTRY_NONE {
            let (id, level) = unpack_entry(entry);
            if id >= count {
                return Err(Error::Corrupt("snapshot: entry out of range".into()));
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
        });
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
        let index = Hnsw::new(HnswConfig::default());
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
}
