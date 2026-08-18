//! The database: vector store + HNSW index + WAL, one directory on disk.
//!
//!   dir/
//!     vectors.store   mmap vector segments (M2)
//!     index.snap      HNSW checkpoint (atomic rename)
//!     pq.codebook     PQ centroids, if trained (atomic rename)
//!     wal             insert log since the last checkpoint
//!
//! Write path: WAL append + fdatasync (the ack point), then graph insert.
//! Recovery: load snapshot if present, then replay WAL records at or past
//! the snapshot watermark — re-appending vectors the store hadn't flushed
//! and re-inserting graph nodes the snapshot didn't cover. Replayed graph
//! inserts resample levels; that changes the graph shape, not its
//! correctness.
//!
//! Locking: searches are lock-free end to end. Inserts share a read lock on
//! `maintenance`; `checkpoint()` takes it exclusively so the snapshot sees
//! no half-linked nodes and can safely reset the WAL.

use std::path::{Path, PathBuf};

use parking_lot::{Mutex, RwLock};
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Deterministic per-id RNG for HNSW level assignment. Seeding from the id
/// (not entropy) makes the graph a pure function of the data and insert
/// order, so recovery rebuilds a byte-identical graph and crash tests are
/// reproducible. The constant just decorrelates the stream from the raw id.
fn level_rng(id: u64) -> SmallRng {
    SmallRng::seed_from_u64(id ^ 0x9E37_79B9_7F4A_7C15)
}

use crate::distance::Metric;
use crate::error::{Error, Result};
use crate::hnsw::{Hnsw, HnswConfig};
use crate::pq::{adc_distance, ProductQuantizer};
use crate::storage::{Neighbor, VectorStore};
use crate::wal::{SyncPolicy, Wal, WalOp};

struct PqState {
    pq: ProductQuantizer,
    /// Code for id `i` at `codes[i*m .. (i+1)*m]`. In-memory in v1 —
    /// rebuilt from the store at `train_pq` time; codes for later inserts
    /// are appended as they arrive.
    codes: Vec<u8>,
}

pub struct Db {
    store: VectorStore,
    index: Hnsw,
    /// `None` disables logging entirely (in-memory / benchmark mode): the
    /// mmap store and graph still work and can be checkpointed, but inserts
    /// are not crash-durable. This is the apples-to-apples configuration for
    /// benchmarking against a pure in-memory library like FAISS.
    wal: Option<Mutex<Wal>>,
    maintenance: RwLock<()>,
    pq: RwLock<Option<PqState>>,
    snap_path: PathBuf,
    pq_path: PathBuf,
}

pub struct DbOptions {
    pub config: HnswConfig,
    pub sync: SyncPolicy,
    /// Enable the write-ahead log (default true). Set false for a pure
    /// in-memory index with no per-insert logging.
    pub enable_wal: bool,
}

impl Default for DbOptions {
    fn default() -> Self {
        Self {
            config: HnswConfig::default(),
            sync: SyncPolicy::Always,
            enable_wal: true,
        }
    }
}

impl Db {
    /// Open or create a database in `dir`.
    pub fn open<P: AsRef<Path>>(dir: P, dim: usize, opts: DbOptions) -> Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let store_path = dir.join("vectors.store");
        let snap_path = dir.join("index.snap");
        let pq_path = dir.join("pq.codebook");
        let wal_path = dir.join("wal");

        let store = if store_path.exists() {
            VectorStore::open(&store_path)?
        } else {
            VectorStore::create(&store_path, dim)?
        };
        if store.dim() != dim {
            return Err(Error::DimensionMismatch {
                expected: store.dim(),
                got: dim,
            });
        }

        let (index, replay_from) = if snap_path.exists() {
            Hnsw::load(&snap_path)?
        } else {
            (Hnsw::new(opts.config), 0)
        };

        let wal = if opts.enable_wal {
            let (mut wal, records) = Wal::open(&wal_path, opts.sync)?;
            wal.ensure_lsn_at_least(replay_from);
            for (lsn, op) in records {
                if lsn < replay_from {
                    continue; // already covered by the snapshot
                }
                let WalOp::Insert { id, vector } = op;
                // Idempotent replay: the store may have flushed past the
                // snapshot, or crashed before flushing anything.
                if (id as usize) == store.len() {
                    store.append(&vector)?;
                } else if (id as usize) > store.len() {
                    return Err(Error::Corrupt(format!(
                        "WAL skips store id {}",
                        store.len()
                    )));
                }
                if (id as usize) >= index.len() {
                    index.insert(&store, id, &mut level_rng(id));
                }
            }
            Some(Mutex::new(wal))
        } else {
            None
        };
        if index.len() != store.len() {
            // Rows flushed by the store but never WAL-acked (crash between
            // mmap writeback and fsync) are unreachable; that is fine — they
            // were never acknowledged. But the index can't exceed the store.
            if index.len() > store.len() {
                return Err(Error::Corrupt("index ahead of vector store".into()));
            }
        }

        // After replay, so the codes we derive cover every recovered vector.
        let pq = if pq_path.exists() {
            let quantizer = load_codebook(&pq_path)?;
            if quantizer.dim() != store.dim() {
                return Err(Error::Corrupt(format!(
                    "pq.codebook is for dim {}, store has {}",
                    quantizer.dim(),
                    store.dim()
                )));
            }
            let codes = encode_all(&quantizer, &store);
            Some(PqState {
                pq: quantizer,
                codes,
            })
        } else {
            None
        };

        Ok(Self {
            store,
            index,
            wal,
            maintenance: RwLock::new(()),
            pq: RwLock::new(pq),
            snap_path,
            pq_path,
        })
    }

    /// Insert a vector. When this returns under `SyncPolicy::Always` (and the
    /// WAL is enabled), the insert is durable. Callable from many threads.
    pub fn insert(&self, vector: &[f32]) -> Result<u64> {
        let _shared = self.maintenance.read();
        let id = match &self.wal {
            Some(wal) => {
                // One critical section orders store ids and WAL records
                // identically; fsync happens inside so an ack implies
                // durability.
                let mut wal = wal.lock();
                let id = self.store.append(vector)?;
                wal.append(&WalOp::Insert {
                    id,
                    vector: vector.to_vec(),
                })?;
                id
            }
            None => self.store.append(vector)?,
        };
        self.index.insert(&self.store, id, &mut level_rng(id));
        if let Some(state) = self.pq.write().as_mut() {
            let m = state.pq.m();
            let need = (id as usize + 1) * m;
            if state.codes.len() < need {
                state.codes.resize(need, 0);
            }
            state
                .pq
                .encode(vector, &mut state.codes[id as usize * m..][..m]);
        }
        Ok(id)
    }

    /// Train product quantization on the current contents (L2 only) and
    /// encode every stored vector. Later inserts are encoded on the fly.
    /// `m` subquantizers of 256 centroids: `dim/m*4 : 1` compression.
    ///
    /// The k-means runs unlocked -- it only needs a sample, and rows already
    /// appended never move -- but encoding and publishing happen together
    /// under the exclusive maintenance lock, so `codes` always covers exactly
    /// the store that `search_pq` will traverse. Concurrent inserts block only
    /// for the encode, not for the training.
    pub fn train_pq(&self, m: usize, iters: usize, max_samples: usize) -> Result<()> {
        if self.index.config().metric != Metric::L2 {
            return Err(Error::Corrupt("PQ supports L2 only in v1".into()));
        }
        let n = self.store.len();
        let dim = self.store.dim();
        if n < crate::pq::K {
            return Err(Error::Corrupt(format!(
                "need >= {} vectors to train PQ",
                crate::pq::K
            )));
        }
        let step = (n / max_samples.max(1)).max(1);
        let mut samples = Vec::with_capacity((n / step + 1) * dim);
        for id in (0..n).step_by(step) {
            samples.extend_from_slice(unsafe { self.store.get_unchecked(id as u64) });
        }
        let mut rng = SmallRng::from_entropy();
        let pq = ProductQuantizer::train(&samples, dim, m, iters, &mut rng);

        // Exclusive: an insert landing between encode_all reading store.len()
        // and this publish would see `pq == None`, write no code, and then be
        // searched against codes that don't describe it.
        let _exclusive = self.maintenance.write();
        let codes = encode_all(&pq, &self.store);
        *self.pq.write() = Some(PqState { pq, codes });
        Ok(())
    }

    /// Ids whose stored PQ code disagrees with a fresh encode of their
    /// vector, plus `(codes covered, store len)`. The invariant `search_pq`
    /// depends on: every id the graph can return has a real code. A missed
    /// insert shows up either as short coverage or as a zero-filled code.
    #[cfg(test)]
    fn pq_inconsistent_ids(&self) -> Option<(Vec<u64>, usize, usize)> {
        let state = self.pq.read();
        let state = state.as_ref()?;
        let m = state.pq.m();
        let covered = state.codes.len() / m;
        let stored = self.store.len();
        let mut bad = Vec::new();
        let mut fresh = vec![0u8; m];
        for id in 0..covered.min(stored) {
            state
                .pq
                .encode(unsafe { self.store.get_unchecked(id as u64) }, &mut fresh);
            if fresh != state.codes[id * m..][..m] {
                bad.push(id as u64);
            }
        }
        Some((bad, covered, stored))
    }

    /// Approximate search over PQ codes (ADC), then re-rank the best
    /// `k * rerank` candidates with full-precision vectors from the store.
    /// `rerank = 0` skips re-ranking and returns raw ADC results.
    pub fn search_pq(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        rerank: usize,
    ) -> Result<Vec<Neighbor>> {
        if query.len() != self.store.dim() {
            return Err(Error::DimensionMismatch {
                expected: self.store.dim(),
                got: query.len(),
            });
        }
        let state = self.pq.read();
        let Some(state) = state.as_ref() else {
            return Err(Error::Corrupt("PQ not trained; call train_pq first".into()));
        };
        let m = state.pq.m();
        let table = state.pq.adc_table(query);
        let fetch = if rerank == 0 { k } else { k * rerank };
        let mut hits = self.index.search_with_oracle(
            |id| adc_distance(&table, &state.codes[id as usize * m..][..m]),
            fetch,
            ef.max(fetch),
        );
        if rerank > 0 {
            let kernel = self.index.config().metric.kernel();
            for h in hits.iter_mut() {
                h.distance = kernel(query, unsafe { self.store.get_unchecked(h.id) });
            }
            hits.sort();
            hits.truncate(k);
        }
        Ok(hits)
    }

    /// Lock-free approximate search.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<Neighbor>> {
        if query.len() != self.store.dim() {
            return Err(Error::DimensionMismatch {
                expected: self.store.dim(),
                got: query.len(),
            });
        }
        Ok(self.index.search(&self.store, query, k, ef))
    }

    /// Snapshot the index, flush the store, and truncate the WAL. Blocks
    /// until in-flight inserts finish; searches keep running throughout.
    /// With the WAL disabled, snapshots the graph at LSN 0.
    pub fn checkpoint(&self) -> Result<()> {
        let _exclusive = self.maintenance.write();
        self.store.flush()?;
        if let Some(state) = self.pq.read().as_ref() {
            save_codebook(&self.pq_path, &state.pq)?;
        }
        match &self.wal {
            Some(wal) => {
                let mut wal = wal.lock();
                self.index.save(&self.snap_path, wal.next_lsn())?;
                wal.reset()?;
            }
            None => self.index.save(&self.snap_path, 0)?,
        }
        Ok(())
    }

    /// Bulk-insert a batch of contiguous vectors (`batch.len() / dim` rows)
    /// in parallel across rayon's thread pool. Ids are assigned densely
    /// starting at the current length; returns the first id. Intended for
    /// benchmark/bulk-load — obeys the WAL setting like `insert`.
    ///
    /// The store is grown up front so worker threads only write disjoint,
    /// already-mapped rows; graph inserts run concurrently through the
    /// lock-free path.
    pub fn add_batch(&self, batch: &[f32]) -> Result<u64> {
        use rayon::prelude::*;
        let dim = self.store.dim();
        assert_eq!(batch.len() % dim, 0, "batch not a multiple of dim");
        let base = self.store.len() as u64;

        // Phase 1: append rows (+ WAL) sequentially to fix the id order.
        for row in batch.chunks_exact(dim) {
            self.insert_row_storage_only(row)?;
        }
        // Phase 2: build graph links in parallel.
        let n = batch.len() / dim;
        (0..n as u64).into_par_iter().for_each(|i| {
            let id = base + i;
            self.index.insert(&self.store, id, &mut level_rng(id));
        });
        if let Some(state) = self.pq.write().as_mut() {
            let PqState { pq, codes } = state;
            let m = pq.m();
            codes.resize((base as usize + n) * m, 0);
            codes[base as usize * m..]
                .par_chunks_mut(m)
                .enumerate()
                .for_each(|(i, out)| {
                    pq.encode(unsafe { self.store.get_unchecked(base + i as u64) }, out);
                });
        }
        Ok(base)
    }

    fn insert_row_storage_only(&self, vector: &[f32]) -> Result<u64> {
        match &self.wal {
            Some(wal) => {
                let mut wal = wal.lock();
                let id = self.store.append(vector)?;
                wal.append(&WalOp::Insert {
                    id,
                    vector: vector.to_vec(),
                })?;
                Ok(id)
            }
            None => self.store.append(vector),
        }
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dim(&self) -> usize {
        self.store.dim()
    }

    pub fn get(&self, id: u64) -> Result<&[f32]> {
        self.store.get(id)
    }

    #[cfg(test)]
    fn index(&self) -> &Hnsw {
        &self.index
    }

    /// Number of layer-0 nodes with zero incoming links (unreachable from
    /// the entry point). Diagnostic for graph health after recovery.
    pub fn orphan_count(&self) -> usize {
        self.index.zero_indegree_count()
    }
}

// --- PQ codebook persistence -------------------------------------------------
//
// Only the *codebook* is stored, never the per-vector codes: codes are a pure
// function of (codebook, store), so re-deriving them on open means they can
// never drift out of sync with the vectors -- including vectors recovered from
// the WAL after the last checkpoint. The codebook is tiny (m*256*sub_dim
// floats, 128 KiB at m=16/dim=128); re-encoding is the cost, and it is
// rayon-parallel.

const PQ_MAGIC: u32 = 0x564B_5051; // "VKPQ"
const PQ_VERSION: u32 = 1;

/// Encode every vector in `store` under `pq`.
fn encode_all(pq: &ProductQuantizer, store: &VectorStore) -> Vec<u8> {
    use rayon::prelude::*;
    let (n, m) = (store.len(), pq.m());
    let mut codes = vec![0u8; n * m];
    codes.par_chunks_mut(m).enumerate().for_each(|(id, out)| {
        pq.encode(unsafe { store.get_unchecked(id as u64) }, out);
    });
    codes
}

/// Write the codebook via temp file + fsync + rename, like the HNSW snapshot,
/// so a crash mid-checkpoint leaves the previous codebook intact.
fn save_codebook(path: &Path, pq: &ProductQuantizer) -> Result<()> {
    use std::io::Write;

    let body = pq.to_bytes();
    let mut buf = Vec::with_capacity(body.len() + 12);
    buf.extend_from_slice(&PQ_MAGIC.to_le_bytes());
    buf.extend_from_slice(&PQ_VERSION.to_le_bytes());
    buf.extend_from_slice(&body);
    let crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    let tmp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(&buf)?;
    f.sync_all()?;
    drop(f);
    std::fs::rename(&tmp, path)?;
    if let Some(dir) = path.parent() {
        std::fs::File::open(dir)?.sync_all()?;
    }
    Ok(())
}

fn load_codebook(path: &Path) -> Result<ProductQuantizer> {
    let raw = std::fs::read(path)?;
    if raw.len() < 16 {
        return Err(Error::Corrupt("pq.codebook truncated".into()));
    }
    let (payload, crc_bytes) = raw.split_at(raw.len() - 4);
    let want = u32::from_le_bytes(crc_bytes.try_into().unwrap());
    if crc32fast::hash(payload) != want {
        return Err(Error::Corrupt("pq.codebook checksum mismatch".into()));
    }
    if u32::from_le_bytes(payload[0..4].try_into().unwrap()) != PQ_MAGIC {
        return Err(Error::Corrupt("pq.codebook bad magic".into()));
    }
    let version = u32::from_le_bytes(payload[4..8].try_into().unwrap());
    if version != PQ_VERSION {
        return Err(Error::Corrupt(format!(
            "pq.codebook version {version}, expected {PQ_VERSION}"
        )));
    }
    ProductQuantizer::from_bytes(&payload[8..])
        .ok_or_else(|| Error::Corrupt("pq.codebook malformed".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_for(id: u64, dim: usize) -> Vec<f32> {
        // Distinct ids must give distinct vectors (ties at distance 0 make
        // top-1 assertions ambiguous), so mix id and j through a 64-bit LCG.
        (0..dim as u64)
            .map(|j| {
                let x = id
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(j.wrapping_mul(1442695040888963407));
                ((x >> 33) as u32) as f32 / u32::MAX as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn train_pq_concurrent_with_inserts_keeps_codes_covering_the_store() {
        // train_pq used to take no maintenance lock at all, while every other
        // mutator does. It snapshotted store.len(), trained (slow), then
        // published codes sized for the *old* length. Inserts landing in that
        // window saw `pq == None`, wrote no code, and were then either indexed
        // out of bounds by search_pq's oracle or silently scored as centroid 0
        // in every subspace.
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use std::sync::{Arc, Barrier};

        const SEED: usize = 60_000;
        let dir = tempfile::tempdir().unwrap();
        let dim = 32;
        let seed: Vec<f32> = (0..SEED * dim)
            .map(|i| ((i * 7919) % 977) as f32 / 977.0)
            .collect();
        let db = Arc::new(Db::open(dir.path(), dim, DbOptions::default()).unwrap());
        db.add_batch(&seed).unwrap();

        // The writer must stay in flight for the *whole* of train_pq, because
        // the window is between encode_all reading store.len() and the publish.
        // A writer that finishes early (400 inserts take ~1ms, training takes
        // seconds) never overlaps it and the bug stays invisible.
        let done = Arc::new(AtomicBool::new(false));
        let written = Arc::new(AtomicU64::new(0));
        let gate = Arc::new(Barrier::new(2));

        let trainer = {
            let (db, gate, done) = (db.clone(), gate.clone(), done.clone());
            std::thread::spawn(move || {
                gate.wait();
                db.train_pq(8, 20, 100_000).unwrap();
                done.store(true, Ordering::SeqCst);
            })
        };
        let writer = {
            let (db, gate, done, written) =
                (db.clone(), gate.clone(), done.clone(), written.clone());
            std::thread::spawn(move || {
                gate.wait();
                let mut i = 0u64;
                while !done.load(Ordering::SeqCst) {
                    db.insert(&vec![(i % 400) as f32 / 400.0; dim]).unwrap();
                    i += 1;
                }
                written.store(i, Ordering::SeqCst);
            })
        };
        trainer.join().unwrap();
        writer.join().unwrap();
        let inserted = written.load(Ordering::SeqCst) as usize;
        assert!(inserted > 0, "writer never overlapped training");

        let (bad, covered, stored) = db.pq_inconsistent_ids().expect("pq trained");
        assert_eq!(stored, SEED + inserted);
        assert_eq!(
            covered, stored,
            "codes must cover every stored vector ({covered} codes vs {stored} vectors)"
        );
        assert!(
            bad.is_empty(),
            "{} of {stored} ids have a code that doesn't match their vector \
             (first few: {:?}) -- inserts were dropped during training",
            bad.len(),
            &bad[..bad.len().min(8)]
        );

        // Every id the graph can return must be scorable: with codes short of
        // the store this panicked inside search_pq's rayon oracle.
        for i in [0u64, (SEED as u64) - 1, SEED as u64, (stored as u64) - 1] {
            let v = db.get(i).unwrap().to_vec();
            assert_eq!(db.search_pq(&v, 10, 128, 4).unwrap().len(), 10);
        }
    }

    #[test]
    fn pq_survives_checkpoint_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let dim = 32;
        let n = 600;
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
            .collect();

        let before = {
            let db = Db::open(dir.path(), dim, DbOptions::default()).unwrap();
            db.add_batch(&data).unwrap();
            db.train_pq(8, 10, 100_000).unwrap();
            let q = &data[13 * dim..][..dim];
            let hits = db.search_pq(q, 10, 64, 4).unwrap();
            db.checkpoint().unwrap();
            hits
        };

        // Reopen: search_pq used to fail here with "PQ not trained", because
        // the codebook was only ever held in memory.
        let db = Db::open(dir.path(), dim, DbOptions::default()).unwrap();
        let q = &data[13 * dim..][..dim];
        let after = db
            .search_pq(q, 10, 64, 4)
            .expect("PQ must be restored from pq.codebook on open");
        let ids = |v: &[crate::storage::Neighbor]| v.iter().map(|h| h.id).collect::<Vec<_>>();
        assert_eq!(
            ids(&before),
            ids(&after),
            "same codebook must rank the same"
        );

        // Codes are re-derived from the store, so vectors inserted after the
        // checkpoint are covered too.
        db.insert(&vec![0.42f32; dim]).unwrap();
        db.search_pq(q, 10, 64, 4).unwrap();
    }

    #[test]
    fn corrupt_codebook_is_rejected_not_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let dim = 32;
        let data: Vec<f32> = (0..600 * dim).map(|i| (i % 251) as f32 / 251.0).collect();
        {
            let db = Db::open(dir.path(), dim, DbOptions::default()).unwrap();
            db.add_batch(&data).unwrap();
            db.train_pq(8, 5, 100_000).unwrap();
            db.checkpoint().unwrap();
        }
        let path = dir.path().join("pq.codebook");
        let mut raw = std::fs::read(&path).unwrap();
        let mid = raw.len() / 2;
        raw[mid] ^= 0xFF;
        std::fs::write(&path, &raw).unwrap();

        match Db::open(dir.path(), dim, DbOptions::default()) {
            Err(e) => assert!(
                e.to_string().contains("checksum mismatch"),
                "unhelpful error: {e}"
            ),
            Ok(_) => panic!("a corrupt codebook must not open silently"),
        }
    }

    #[test]
    fn insert_search_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let dim = 16;
        {
            let db = Db::open(dir.path(), dim, DbOptions::default()).unwrap();
            for i in 0..500 {
                assert_eq!(db.insert(&vec_for(i, dim)).unwrap(), i);
            }
            let hits = db.search(&vec_for(123, dim), 1, 64).unwrap();
            assert_eq!(hits[0].id, 123);
            assert!(hits[0].distance < 1e-6);
        } // dropped without checkpoint: recovery is pure WAL replay
        let db = Db::open(dir.path(), dim, DbOptions::default()).unwrap();
        assert_eq!(db.len(), 500);
        for i in (0..500).step_by(41) {
            assert_eq!(db.get(i).unwrap(), vec_for(i, dim).as_slice());
            let hits = db.search(&vec_for(i, dim), 1, 64).unwrap();
            assert_eq!(hits[0].id, i, "id {i} lost after reopen");
        }
    }

    #[test]
    fn checkpoint_then_more_inserts_then_recover() {
        let dir = tempfile::tempdir().unwrap();
        let dim = 8;
        {
            let db = Db::open(dir.path(), dim, DbOptions::default()).unwrap();
            for i in 0..300 {
                db.insert(&vec_for(i, dim)).unwrap();
            }
            db.checkpoint().unwrap();
            for i in 300..400 {
                db.insert(&vec_for(i, dim)).unwrap();
            }
        }
        let db = Db::open(dir.path(), dim, DbOptions::default()).unwrap();
        assert_eq!(db.len(), 400);
        for i in [0, 299, 300, 399] {
            let hits = db.search(&vec_for(i, dim), 1, 64).unwrap();
            assert_eq!(hits[0].id, i);
        }
    }

    #[test]
    fn lsn_survives_checkpoint_then_crash() {
        // Regression: checkpoint resets the WAL; if the LSN counter restarts
        // at 0 on reopen, records written after recovery sit below the
        // snapshot watermark and the NEXT recovery silently drops them.
        let dir = tempfile::tempdir().unwrap();
        let dim = 8;
        {
            let db = Db::open(dir.path(), dim, DbOptions::default()).unwrap();
            for i in 0..300 {
                db.insert(&vec_for(i, dim)).unwrap();
            }
            db.checkpoint().unwrap();
        } // crash #1: empty WAL, snapshot watermark = 300
        {
            let db = Db::open(dir.path(), dim, DbOptions::default()).unwrap();
            for i in 300..400 {
                db.insert(&vec_for(i, dim)).unwrap();
            }
        } // crash #2: no checkpoint — those 100 inserts live only in the WAL
        let db = Db::open(dir.path(), dim, DbOptions::default()).unwrap();
        assert_eq!(
            db.len(),
            400,
            "post-checkpoint inserts lost on second recovery"
        );
        let hits = db.search(&vec_for(399, dim), 1, 64).unwrap();
        assert_eq!(hits[0].id, 399);
    }

    #[test]
    fn pq_search_with_rerank_recovers_exact_ranking() {
        let dir = tempfile::tempdir().unwrap();
        let dim = 32;
        let db = Db::open(
            dir.path(),
            dim,
            DbOptions {
                sync: crate::wal::SyncPolicy::Never,
                ..Default::default()
            },
        )
        .unwrap();
        // Clustered data so PQ codebooks have structure to capture.
        let mut rng = rand::rngs::SmallRng::seed_from_u64(6);
        use rand::Rng;
        let centers: Vec<Vec<f32>> = (0..8)
            .map(|_| (0..dim).map(|_| rng.gen_range(-4.0f32..4.0)).collect())
            .collect();
        let sample = |rng: &mut rand::rngs::SmallRng| -> Vec<f32> {
            let c = &centers[rng.gen_range(0..centers.len())];
            c.iter().map(|x| x + rng.gen_range(-0.5..0.5)).collect()
        };
        for _ in 0..5000 {
            let v = sample(&mut rng);
            db.insert(&v).unwrap();
        }
        db.train_pq(8, 10, 5000).unwrap();
        // Inserts after training must be searchable too.
        for _ in 0..500 {
            let v = sample(&mut rng);
            db.insert(&v).unwrap();
        }

        let mut hits_pq = 0usize;
        let mut hits_raw = 0usize;
        let queries = 50;
        for _ in 0..queries {
            let q = sample(&mut rng);
            let truth: std::collections::HashSet<u64> =
                crate::storage::exact_search(&db.store, &q, 10, Metric::L2)
                    .into_iter()
                    .map(|n| n.id)
                    .collect();
            let reranked = db.search_pq(&q, 10, 128, 4).unwrap();
            let raw = db.search_pq(&q, 10, 128, 0).unwrap();
            hits_pq += reranked.iter().filter(|n| truth.contains(&n.id)).count();
            hits_raw += raw.iter().filter(|n| truth.contains(&n.id)).count();
        }
        let recall_pq = hits_pq as f64 / (queries * 10) as f64;
        let recall_raw = hits_raw as f64 / (queries * 10) as f64;
        assert!(recall_pq >= 0.85, "PQ+rerank recall@10 = {recall_pq}");
        assert!(recall_pq >= recall_raw, "re-ranking should not hurt recall");
    }

    #[test]
    fn no_wal_batch_build_matches_sequential() {
        // In-memory (no WAL) parallel batch build must produce a working
        // index: every batch vector is its own nearest neighbor.
        let dir = tempfile::tempdir().unwrap();
        let dim = 24;
        let db = Db::open(
            dir.path(),
            dim,
            DbOptions {
                enable_wal: false,
                ..Default::default()
            },
        )
        .unwrap();
        let n = 5000;
        let mut batch = Vec::with_capacity(n * dim);
        for i in 0..n as u64 {
            batch.extend_from_slice(&vec_for(i, dim));
        }
        let base = db.add_batch(&batch).unwrap();
        assert_eq!(base, 0);
        assert_eq!(db.len(), n);
        // No WAL file should exist.
        assert!(!dir.path().join("wal").exists());
        for id in (0..n as u64).step_by(53) {
            let hits = db.search(&vec_for(id, dim), 1, 64).unwrap();
            assert_eq!(hits[0].id, id, "batch node {id} not found");
        }
        // Orphan rate stays negligible under the parallel build.
        assert!(db.orphan_count() as f64 / db.len() as f64 <= 0.001);
    }

    #[test]
    fn recovery_preserves_graph_connectivity() {
        // Mimics the crash harness deterministically: insert, checkpoint,
        // insert more, drop (no final checkpoint => WAL replay on reopen),
        // repeated. A clean build has zero unreachable nodes, so the
        // recovered graph must too.
        let dir = tempfile::tempdir().unwrap();
        let dim = 16;
        let mut done = 0u64;
        let batches = [2000u64, 1500, 3000, 1200, 2500];
        for (b, &count) in batches.iter().enumerate() {
            let db = Db::open(dir.path(), dim, DbOptions::default()).unwrap();
            assert_eq!(db.len() as u64, done, "reopened at wrong length");
            for i in done..done + count {
                db.insert(&vec_for(i, dim)).unwrap();
            }
            done += count;
            // Checkpoint on some cycles, drop-without-checkpoint on others.
            if b % 2 == 0 {
                db.checkpoint().unwrap();
            }
            // Deterministic per-id levels => recovery rebuilds an identical
            // graph, so its orphan rate matches a clean build's (negligible).
            let orphans = db.index().zero_indegree_nodes();
            let rate = orphans.len() as f64 / db.len() as f64;
            assert!(
                rate <= 0.001,
                "cycle {b}: recovery inflated unreachable rate to {rate} ({} nodes)",
                orphans.len()
            );
        }
        assert_eq!(done, batches.iter().sum());
    }

    #[test]
    fn concurrent_inserts_with_checkpoints() {
        let dir = tempfile::tempdir().unwrap();
        let dim = 8;
        let db = Db::open(dir.path(), dim, DbOptions::default()).unwrap();
        std::thread::scope(|s| {
            for w in 0..4u64 {
                let db = &db;
                s.spawn(move || {
                    for i in 0..250u64 {
                        db.insert(&vec_for(w * 1000 + i, dim)).unwrap();
                    }
                });
            }
            let db = &db;
            s.spawn(move || {
                for _ in 0..5 {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    db.checkpoint().unwrap();
                }
            });
        });
        assert_eq!(db.len(), 1000);
        drop(db);
        let db = Db::open(dir.path(), dim, DbOptions::default()).unwrap();
        assert_eq!(db.len(), 1000);
    }
}
