//! Brute-force exact k-NN over a `VectorStore`.
//!
//! This is the recall oracle: HNSW and PQ results are always scored against
//! what this returns. Parallelized with rayon — each chunk keeps a local
//! top-k max-heap, then the partial heaps merge.

use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::distance::Metric;
use crate::storage::VectorStore;

/// A scored candidate. Ordered so that a `BinaryHeap` is a max-heap on
/// distance (worst candidate on top), which is what bounded top-k wants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: u64,
    pub distance: f32,
}

impl Eq for Neighbor {}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Total order via total_cmp; ties broken by id for determinism.
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

/// Push into a heap bounded to `k` elements (max-heap: root is the worst).
#[inline]
pub fn push_bounded(heap: &mut BinaryHeap<Neighbor>, item: Neighbor, k: usize) {
    if heap.len() < k {
        heap.push(item);
    } else if let Some(worst) = heap.peek() {
        if item < *worst {
            heap.pop();
            heap.push(item);
        }
    }
}

/// Exact k nearest neighbors of `query`, sorted ascending by distance.
pub fn search(store: &VectorStore, query: &[f32], k: usize, metric: Metric) -> Vec<Neighbor> {
    let n = store.len() as u64;
    let kernel = metric.kernel();
    let chunk = 16_384u64;

    let mut merged: BinaryHeap<Neighbor> = (0..n)
        .step_by(chunk as usize)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|start| {
            let end = (start + chunk).min(n);
            let mut heap = BinaryHeap::with_capacity(k + 1);
            for id in start..end {
                let v = unsafe { store.get_unchecked(id) };
                push_bounded(&mut heap, Neighbor { id, distance: kernel(query, v) }, k);
            }
            heap
        })
        .reduce(BinaryHeap::new, |mut a, b| {
            for item in b {
                push_bounded(&mut a, item, k);
            }
            a
        });

    let mut out = Vec::with_capacity(merged.len());
    while let Some(item) = merged.pop() {
        out.push(item);
    }
    out.reverse();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn matches_naive_search() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let dim = 24;
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::create(dir.path().join("v.store"), dim).unwrap();
        let mut all: Vec<Vec<f32>> = Vec::new();
        for _ in 0..2000 {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            store.append(&v).unwrap();
            all.push(v);
        }
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let kernel = metric.kernel();
            let mut naive: Vec<Neighbor> = all
                .iter()
                .enumerate()
                .map(|(i, v)| Neighbor { id: i as u64, distance: kernel(&query, v) })
                .collect();
            naive.sort();
            naive.truncate(10);

            let got = search(&store, &query, 10, metric);
            assert_eq!(got, naive, "metric {metric:?}");
        }
    }

    #[test]
    fn k_larger_than_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::create(dir.path().join("v.store"), 4).unwrap();
        for i in 0..3 {
            store.append(&[i as f32; 4]).unwrap();
        }
        let got = search(&store, &[0.0; 4], 10, Metric::L2);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].id, 0);
    }
}
