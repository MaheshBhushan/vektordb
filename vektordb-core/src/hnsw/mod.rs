//! HNSW graph index, implemented from Malkov & Yashunin,
//! "Efficient and robust approximate nearest neighbor search using
//! Hierarchical Navigable Small World graphs" (2016).
//!
//! Algorithm numbering in comments refers to the paper:
//!   Alg. 1  INSERT
//!   Alg. 2  SEARCH-LAYER
//!   Alg. 4  SELECT-NEIGHBORS-HEURISTIC (extendCandidates=false,
//!           keepPrunedConnections=true)
//!   Alg. 5  K-NN-SEARCH
//!
//! # Concurrency model (M4)
//!
//! Searches take no locks, ever:
//! - Each node's per-layer adjacency is an immutable `Vec<u64>` behind a
//!   `crossbeam_epoch::Atomic` pointer. Readers pin an epoch guard and
//!   acquire-load the pointer; writers build a replacement list, publish it
//!   with a release store, and retire the old block through the epoch
//!   collector so in-flight readers keep a valid snapshot (RCU).
//! - Nodes live in an append-only `SegVec` with stable addresses; a node id
//!   only reaches readers through a release-published link or the entry
//!   pointer, which happens after the node and its vector are fully written.
//!
//! Inserts run concurrently with each other, coordinated by a per-node
//! mutex that serializes rewrites of that node's adjacency only. As in
//! hnswlib, two racing inserts can each miss the other's just-added node in
//! their construction searches — the graph is still valid, and the effect
//! is a marginal recall cost, measured (not hidden) by the stress test.

mod persist;
mod scratch;
mod segvec;

use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering};

use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned};
use parking_lot::Mutex;
use rand::Rng;

use crate::distance::{DistanceFn, Metric};
use crate::storage::exact::push_bounded;
use crate::storage::{Neighbor, VectorStore};

use scratch::VisitedSet;
use segvec::SegVec;

#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    /// Max links per node on layers > 0. Layer 0 gets `2 * m`.
    pub m: usize,
    pub ef_construction: usize,
    pub metric: Metric,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            metric: Metric::L2,
        }
    }
}

/// Entry pointer packed into one atomic: high 8 bits level, low 56 bits id.
const ENTRY_NONE: u64 = u64::MAX;

#[inline]
fn pack_entry(id: u64, level: usize) -> u64 {
    ((level as u64) << 56) | id
}

#[inline]
fn unpack_entry(packed: u64) -> (u64, usize) {
    (packed & ((1 << 56) - 1), (packed >> 56) as usize)
}

struct Node {
    /// Adjacency per layer, `0..=level`. Null pointer = empty list.
    links: Box<[Atomic<Vec<u64>>]>,
    /// Serializes writers rewriting this node's adjacency.
    lock: Mutex<()>,
}

impl Node {
    fn new(level: usize) -> Self {
        Self {
            links: (0..=level).map(|_| Atomic::null()).collect(),
            lock: Mutex::new(()),
        }
    }

    fn level(&self) -> usize {
        self.links.len() - 1
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        // The index is being torn down; no guards can be outstanding.
        let guard = unsafe { epoch::unprotected() };
        for link in self.links.iter() {
            let p = link.load(Ordering::Relaxed, guard);
            if !p.is_null() {
                drop(unsafe { p.into_owned() });
            }
        }
    }
}

thread_local! {
    static VISITED: RefCell<VisitedSet> = RefCell::new(VisitedSet::new());
}

pub struct Hnsw {
    config: HnswConfig,
    kernel: DistanceFn,
    /// Level-generation factor mL = 1/ln(M) (paper, Alg. 1 line 4).
    ml: f64,
    nodes: SegVec<Node>,
    entry: AtomicU64,
    /// Taken only to move the entry point / raise the max level (rare).
    entry_lock: Mutex<()>,
}

impl Hnsw {
    pub fn new(config: HnswConfig) -> Self {
        assert!(config.m >= 2, "M must be at least 2");
        Self {
            kernel: config.metric.kernel(),
            ml: 1.0 / (config.m as f64).ln(),
            config,
            nodes: SegVec::new(),
            entry: AtomicU64::new(ENTRY_NONE),
            entry_lock: Mutex::new(()),
        }
    }

    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.len() == 0
    }

    fn max_degree(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m * 2
        } else {
            self.config.m
        }
    }

    #[inline]
    fn distance(&self, store: &VectorStore, q: &[f32], id: u64) -> f32 {
        (self.kernel)(q, unsafe { store.get_unchecked(id) })
    }

    #[inline]
    fn node(&self, id: u64) -> &Node {
        unsafe { self.nodes.get_unchecked(id as usize) }
    }

    /// Run `f` over the current adjacency snapshot of `id` at `layer`.
    #[inline]
    fn with_links<F: FnMut(u64)>(&self, guard: &Guard, id: u64, layer: usize, mut f: F) {
        let p = self.node(id).links[layer].load(Ordering::Acquire, guard);
        if !p.is_null() {
            for &nb in unsafe { p.deref() }.iter() {
                f(nb);
            }
        }
    }

    /// Alg. 2 SEARCH-LAYER: beam search with `ef` candidates on one layer.
    /// Returns the ef closest found, as a max-heap (worst on top).
    /// Generic over the distance oracle so the same traversal runs on full
    /// vectors (`store`) or PQ codes (ADC table).
    fn search_layer<D: Fn(u64) -> f32>(
        &self,
        guard: &Guard,
        dist: &D,
        visited: &mut VisitedSet,
        entry: u64,
        ef: usize,
        layer: usize,
    ) -> BinaryHeap<Neighbor> {
        visited.begin_pass(self.nodes.len());
        visited.mark(entry);

        let entry_dist = dist(entry);
        let mut candidates = BinaryHeap::new(); // min-heap via Reverse
        candidates.push(std::cmp::Reverse(Neighbor {
            id: entry,
            distance: entry_dist,
        }));
        let mut results: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(ef + 1);
        results.push(Neighbor {
            id: entry,
            distance: entry_dist,
        });

        while let Some(std::cmp::Reverse(current)) = candidates.pop() {
            // Paper line 8: stop once the closest open candidate is farther
            // than the worst of the ef results collected so far.
            let worst = results.peek().map(|n| n.distance).unwrap_or(f32::INFINITY);
            if current.distance > worst && results.len() >= ef {
                break;
            }
            self.with_links(guard, current.id, layer, |next| {
                if visited.mark(next) {
                    return; // already seen this pass
                }
                let d = dist(next);
                let worst = results.peek().map(|n| n.distance).unwrap_or(f32::INFINITY);
                if results.len() < ef || d < worst {
                    candidates.push(std::cmp::Reverse(Neighbor {
                        id: next,
                        distance: d,
                    }));
                    push_bounded(
                        &mut results,
                        Neighbor {
                            id: next,
                            distance: d,
                        },
                        ef,
                    );
                }
            });
        }
        results
    }

    /// Greedy descent used on layers above the target (Alg. 5 lines 2-4 /
    /// Alg. 1 lines 5-7): ef=1 walk to a local minimum.
    fn greedy_descend<D: Fn(u64) -> f32>(
        &self,
        guard: &Guard,
        dist: &D,
        mut entry: u64,
        mut best: f32,
        layer: usize,
    ) -> (u64, f32) {
        loop {
            let mut improved = false;
            self.with_links(guard, entry, layer, |next| {
                let d = dist(next);
                if d < best {
                    entry = next;
                    best = d;
                    improved = true;
                }
            });
            if !improved {
                return (entry, best);
            }
        }
    }

    /// Alg. 4 SELECT-NEIGHBORS-HEURISTIC with keepPrunedConnections.
    /// `candidates` must be sorted ascending by distance to the base point.
    /// A candidate is kept only if it is closer to the base than to every
    /// already-selected neighbor — this spreads links directionally, which
    /// the paper shows is what preserves recall on clustered data.
    fn select_neighbors(&self, store: &VectorStore, candidates: &[Neighbor], m: usize) -> Vec<u64> {
        let mut selected: Vec<Neighbor> = Vec::with_capacity(m);
        let mut pruned: Vec<Neighbor> = Vec::new();

        for &cand in candidates {
            if selected.len() >= m {
                break;
            }
            let cand_vec = unsafe { store.get_unchecked(cand.id) };
            let dominated = selected.iter().any(|s| {
                let d = (self.kernel)(cand_vec, unsafe { store.get_unchecked(s.id) });
                d < cand.distance
            });
            if dominated {
                pruned.push(cand);
            } else {
                selected.push(cand);
            }
        }
        // keepPrunedConnections: top up with the best pruned candidates.
        let mut out: Vec<u64> = selected.iter().map(|n| n.id).collect();
        for p in pruned {
            if out.len() >= m {
                break;
            }
            out.push(p.id);
        }
        out
    }

    /// Replace `id`'s adjacency at `layer` under its writer lock. `mutate`
    /// receives the current snapshot and returns the replacement list.
    fn rewrite_links<F>(&self, guard: &Guard, id: u64, layer: usize, mutate: F)
    where
        F: FnOnce(&[u64]) -> Vec<u64>,
    {
        let node = self.node(id);
        let _w = node.lock.lock();
        let cur = node.links[layer].load(Ordering::Acquire, guard);
        let snapshot: &[u64] = if cur.is_null() {
            &[]
        } else {
            unsafe { cur.deref() }
        };
        let next = mutate(snapshot);
        node.links[layer].store(Owned::new(next), Ordering::Release);
        if !cur.is_null() {
            unsafe { guard.defer_destroy(cur) };
        }
    }

    /// Alg. 1 INSERT. The vector for `id` must already be in `store`.
    /// Safe to call from many threads with distinct ids.
    pub fn insert<R: Rng>(&self, store: &VectorStore, id: u64, rng: &mut R) {
        let level = (-rng.gen::<f64>().ln() * self.ml).floor() as usize;
        let query: &[f32] = unsafe { store.get_unchecked(id) };
        let guard = epoch::pin();

        self.nodes.set(id as usize, Node::new(level));

        // First node: claim the entry point.
        if self.entry.load(Ordering::Acquire) == ENTRY_NONE {
            let _e = self.entry_lock.lock();
            if self.entry.load(Ordering::Relaxed) == ENTRY_NONE {
                self.entry.store(pack_entry(id, level), Ordering::Release);
                return;
            }
        }

        let dist = |id: u64| self.distance(store, query, id);
        let (entry_id, top) = unpack_entry(self.entry.load(Ordering::Acquire));
        let mut ep = entry_id;
        let mut ep_dist = dist(ep);

        // Zoom in through layers above the new node's level.
        for layer in ((level + 1)..=top).rev() {
            (ep, ep_dist) = self.greedy_descend(&guard, &dist, ep, ep_dist, layer);
        }

        // Connect on each layer from min(level, top) down to 0.
        VISITED.with(|v| {
            let visited = &mut *v.borrow_mut();
            for layer in (0..=level.min(top)).rev() {
                let found = self.search_layer(
                    &guard,
                    &dist,
                    visited,
                    ep,
                    self.config.ef_construction,
                    layer,
                );
                let mut candidates = found.into_sorted_vec();
                // Best candidate seeds the next (lower) layer's search.
                if let Some(best) = candidates.first() {
                    ep = best.id;
                }
                candidates.retain(|n| n.id != id);

                // Paper connects M (not M0) links during insert on every layer.
                let chosen = self.select_neighbors(store, &candidates, self.config.m);

                self.rewrite_links(&guard, id, layer, |cur| {
                    let mut list = cur.to_vec();
                    list.extend_from_slice(&chosen);
                    list.dedup();
                    list
                });

                for &nb in &chosen {
                    let max = self.max_degree(layer);
                    self.rewrite_links(&guard, nb, layer, |cur| {
                        if cur.contains(&id) {
                            return cur.to_vec();
                        }
                        let mut list = cur.to_vec();
                        list.push(id);
                        if list.len() <= max {
                            return list;
                        }
                        // Alg. 1 lines 15-17: over-full — re-select with the
                        // heuristic among all current links.
                        let base = unsafe { store.get_unchecked(nb) };
                        let mut cands: Vec<Neighbor> = list
                            .iter()
                            .map(|&x| Neighbor {
                                id: x,
                                distance: (self.kernel)(base, unsafe { store.get_unchecked(x) }),
                            })
                            .collect();
                        cands.sort();
                        self.select_neighbors(store, &cands, max)
                    });
                }
            }
        });

        // Became the highest node: move the entry point.
        if level > top {
            let _e = self.entry_lock.lock();
            let (_, cur_top) = unpack_entry(self.entry.load(Ordering::Relaxed));
            if level > cur_top {
                self.entry.store(pack_entry(id, level), Ordering::Release);
            }
        }
    }

    /// Alg. 5 K-NN-SEARCH. Lock-free; safe to call concurrently with inserts.
    pub fn search(&self, store: &VectorStore, query: &[f32], k: usize, ef: usize) -> Vec<Neighbor> {
        self.search_with_oracle(|id| self.distance(store, query, id), k, ef)
    }

    /// K-NN-SEARCH over an arbitrary distance oracle (e.g. PQ/ADC codes).
    /// The graph shape is shared; only the metric evaluation changes.
    pub fn search_with_oracle<D: Fn(u64) -> f32>(
        &self,
        dist: D,
        k: usize,
        ef: usize,
    ) -> Vec<Neighbor> {
        let packed = self.entry.load(Ordering::Acquire);
        if packed == ENTRY_NONE {
            return Vec::new();
        }
        let (entry_id, top) = unpack_entry(packed);
        let ef = ef.max(k);
        let guard = epoch::pin();

        let mut ep = entry_id;
        let mut ep_dist = dist(ep);
        for layer in (1..=top).rev() {
            (ep, ep_dist) = self.greedy_descend(&guard, &dist, ep, ep_dist, layer);
        }
        let results =
            VISITED.with(|v| self.search_layer(&guard, &dist, &mut v.borrow_mut(), ep, ef, 0));
        let mut out = results.into_sorted_vec();
        out.truncate(k);
        out
    }

    /// Count nodes with zero incoming links on layer 0 (excluding the entry
    /// point). Such a node is unreachable from the entry point at any `ef` —
    /// the classic HNSW "unreachable point" property. Used by tests to
    /// distinguish that from recovery corruption.
    #[cfg(test)]
    pub(crate) fn zero_indegree_nodes(&self) -> Vec<u64> {
        let guard = epoch::pin();
        let n = self.nodes.len();
        let mut indeg = vec![0u32; n];
        for i in 0..n as u64 {
            self.with_links(&guard, i, 0, |nb| indeg[nb as usize] += 1);
        }
        let (entry, _) = unpack_entry(self.entry.load(Ordering::Acquire));
        (0..n as u64)
            .filter(|&i| indeg[i as usize] == 0 && i != entry)
            .collect()
    }

    /// Count of layer-0 nodes with zero in-links (excluding the entry
    /// point) — provably unreachable from the entry point at any `ef`.
    pub fn zero_indegree_count(&self) -> usize {
        let guard = epoch::pin();
        let n = self.nodes.len();
        let mut indeg = vec![0u32; n];
        for i in 0..n as u64 {
            self.with_links(&guard, i, 0, |nb| indeg[nb as usize] += 1);
        }
        let (entry, _) = unpack_entry(self.entry.load(Ordering::Acquire));
        (0..n as u64)
            .filter(|&i| indeg[i as usize] == 0 && i != entry)
            .count()
    }

    /// Structural invariants, checked by tests after building.
    #[cfg(test)]
    fn check_invariants(&self) {
        let guard = epoch::pin();
        for i in 0..self.nodes.len() {
            let node = self.node(i as u64);
            for layer in 0..=node.level() {
                let mut degree = 0;
                self.with_links(&guard, i as u64, layer, |nb| {
                    degree += 1;
                    assert_ne!(nb, i as u64, "self-link at node {i}");
                    assert!(
                        self.node(nb).level() >= layer,
                        "node {i} links to {nb} which does not exist on layer {layer}"
                    );
                });
                assert!(
                    degree <= self.max_degree(layer),
                    "node {i} layer {layer} degree {degree} > max {}",
                    self.max_degree(layer)
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::exact_search;
    use rand::{Rng, SeedableRng};

    /// Sample a point from the blob mixture: cluster center + noise. Data
    /// and queries both come from here — like real ANN benchmarks, queries
    /// are held-out draws from the data distribution (out-of-distribution
    /// queries in high dim are nearly equidistant to everything and are not
    /// what recall targets are calibrated on).
    fn sample_blob(centers: &[Vec<f32>], rng: &mut impl Rng) -> Vec<f32> {
        let c = &centers[rng.gen_range(0..centers.len())];
        c.iter().map(|x| x + rng.gen_range(-1.0..1.0f32)).collect()
    }

    fn make_centers(dim: usize, rng: &mut impl Rng) -> Vec<Vec<f32>> {
        (0..10)
            .map(|_| (0..dim).map(|_| rng.gen_range(-5.0..5.0)).collect())
            .collect()
    }

    fn build_random(
        n: usize,
        dim: usize,
        config: HnswConfig,
        seed: u64,
    ) -> (
        VectorStore,
        Hnsw,
        rand::rngs::StdRng,
        Vec<Vec<f32>>,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::create(dir.path().join("v.store"), dim).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let index = Hnsw::new(config);
        let centers = make_centers(dim, &mut rng);
        for _ in 0..n {
            let v = sample_blob(&centers, &mut rng);
            let id = store.append(&v).unwrap();
            index.insert(&store, id, &mut rng);
        }
        (store, index, rng, centers, dir)
    }

    fn recall_at_k(
        store: &VectorStore,
        index: &Hnsw,
        centers: &[Vec<f32>],
        rng: &mut impl Rng,
        queries: usize,
        k: usize,
        ef: usize,
    ) -> f64 {
        let mut hits = 0usize;
        for _ in 0..queries {
            let q = sample_blob(centers, rng);
            let truth = exact_search(store, &q, k, index.config().metric);
            let got = index.search(store, &q, k, ef);
            let truth_ids: std::collections::HashSet<u64> = truth.iter().map(|n| n.id).collect();
            hits += got.iter().filter(|n| truth_ids.contains(&n.id)).count();
        }
        hits as f64 / (queries * k) as f64
    }

    #[test]
    fn exact_on_tiny_graph() {
        let (store, index, _, _, _dir) = build_random(50, 8, HnswConfig::default(), 3);
        index.check_invariants();
        let truth = exact_search(&store, store.get(0).unwrap(), 1, Metric::L2);
        let got = index.search(&store, store.get(0).unwrap(), 1, 64);
        assert_eq!(got[0].id, truth[0].id);
        assert_eq!(got[0].id, 0);
    }

    #[test]
    fn recall_on_50k_clustered() {
        let (store, index, mut rng, centers, _dir) = build_random(
            50_000,
            32,
            HnswConfig {
                m: 16,
                ef_construction: 200,
                metric: Metric::L2,
            },
            42,
        );
        index.check_invariants();
        let recall = recall_at_k(&store, &index, &centers, &mut rng, 100, 10, 128);
        assert!(recall >= 0.95, "recall@10 = {recall}, expected >= 0.95");
    }

    #[test]
    fn recall_improves_with_ef() {
        let (store, index, mut rng, centers, _dir) =
            build_random(10_000, 16, HnswConfig::default(), 11);
        let low = recall_at_k(&store, &index, &centers, &mut rng, 50, 10, 16);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(999);
        let high = recall_at_k(&store, &index, &centers, &mut rng2, 50, 10, 256);
        assert!(high >= low, "ef=256 recall {high} < ef=16 recall {low}");
        assert!(high >= 0.97, "high-ef recall {high} too low");
    }

    /// Characterize the "unreachable point" property on a clean, never-
    /// crashed, single-threaded build. If this finds zero-indegree nodes,
    /// then the crash harness seeing the same is an HNSW property, not a
    /// recovery bug — and the harness must test durability, not reachability.
    #[test]
    fn clean_build_may_have_unreachable_points() {
        let dim = 16;
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::create(dir.path().join("v.store"), dim).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(12321);
        // Small M makes in-link starvation more likely — same regime the
        // crash harness runs (M defaults to 16, but pruning still bites).
        let index = Hnsw::new(HnswConfig {
            m: 16,
            ef_construction: 200,
            metric: Metric::L2,
        });
        let centers = make_centers(dim, &mut rng);
        for _ in 0..12_000 {
            let v = sample_blob(&centers, &mut rng);
            let id = store.append(&v).unwrap();
            index.insert(&store, id, &mut rng);
        }
        let orphans = index.zero_indegree_nodes();
        // Any orphan is provably unreachable at every ef — a self-query for
        // it cannot return it as top-1. Confirm the two notions agree.
        for &id in orphans.iter().take(5) {
            let hits = index.search(&store, unsafe { store.get_unchecked(id) }, 1, 8192);
            assert_ne!(
                hits.first().map(|h| h.id),
                Some(id),
                "node {id} has zero in-links yet was found — contradiction"
            );
        }
        // This is a documented HNSW property; we only assert it stays rare.
        let frac = orphans.len() as f64 / index.len() as f64;
        assert!(frac < 0.01, "unreachable fraction {frac} unexpectedly high");
        eprintln!(
            "clean build: {} / {} nodes unreachable ({:.4}%)",
            orphans.len(),
            index.len(),
            frac * 100.0
        );
    }

    #[test]
    #[ignore = "diagnostic: sweeps seeds to characterize the orphan rate"]
    fn orphan_rate_across_seeds() {
        let dim = 16;
        let mut total_orphans = 0usize;
        let mut builds_with_orphans = 0usize;
        let n = 12_000;
        let trials = 20;
        for seed in 0..trials as u64 {
            let dir = tempfile::tempdir().unwrap();
            let store = VectorStore::create(dir.path().join("v.store"), dim).unwrap();
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let index = Hnsw::new(HnswConfig::default());
            let centers = make_centers(dim, &mut rng);
            for _ in 0..n {
                let v = sample_blob(&centers, &mut rng);
                let id = store.append(&v).unwrap();
                index.insert(&store, id, &mut rng);
            }
            let o = index.zero_indegree_nodes().len();
            if o > 0 {
                builds_with_orphans += 1;
            }
            total_orphans += o;
        }
        eprintln!(
            "orphans across {trials} clean builds of {n}: total={total_orphans}, \
             builds_with_any={builds_with_orphans}"
        );
    }

    #[test]
    fn empty_and_single() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::create(dir.path().join("v.store"), 4).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let index = Hnsw::new(HnswConfig::default());
        assert!(index.search(&store, &[0.0; 4], 5, 32).is_empty());
        let id = store.append(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        index.insert(&store, id, &mut rng);
        let got = index.search(&store, &[1.0, 2.0, 3.0, 4.0], 5, 32);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, 0);
    }

    /// M4 stress: concurrent inserts + concurrent lock-free searches, then
    /// verify the finished graph both structurally and by recall.
    #[test]
    fn concurrent_insert_and_search() {
        let dim = 16;
        let n_total = 20_000;
        let n_writers = 4;
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::create(dir.path().join("v.store"), dim).unwrap();
        let index = Hnsw::new(HnswConfig {
            m: 12,
            ef_construction: 100,
            metric: Metric::L2,
        });
        let mut seed_rng = rand::rngs::StdRng::seed_from_u64(77);
        let centers = make_centers(dim, &mut seed_rng);

        let stop = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|s| {
            for w in 0..n_writers {
                let (store, index, centers) = (&store, &index, &centers);
                s.spawn(move || {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(1000 + w as u64);
                    for _ in 0..(n_total / n_writers) {
                        let v = sample_blob(centers, &mut rng);
                        let id = store.append(&v).unwrap();
                        index.insert(store, id, &mut rng);
                    }
                });
            }
            for r in 0..2 {
                let (store, index, centers, stop) = (&store, &index, &centers, &stop);
                s.spawn(move || {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(2000 + r as u64);
                    let mut searches = 0usize;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        let q = sample_blob(centers, &mut rng);
                        let got = index.search(store, &q, 10, 64);
                        // Sanity on every result while writers are racing.
                        for n in &got {
                            assert!((n.id as usize) < store.len());
                            assert!(n.distance.is_finite());
                        }
                        searches += 1;
                    }
                    assert!(searches > 0);
                });
            }
            // Writers finish first; then release the readers.
            // (Scoped threads join automatically; signal stop from a monitor.)
            let stop_ref = &stop;
            let index_ref = &index;
            s.spawn(move || {
                while index_ref.len() < n_total {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                stop_ref.store(true, std::sync::atomic::Ordering::Relaxed);
            });
        });

        assert_eq!(index.len(), n_total);
        index.check_invariants();
        let mut rng = rand::rngs::StdRng::seed_from_u64(31337);
        let recall = recall_at_k(&store, &index, &centers, &mut rng, 100, 10, 128);
        assert!(recall >= 0.90, "post-concurrent-build recall@10 = {recall}");
    }
}
