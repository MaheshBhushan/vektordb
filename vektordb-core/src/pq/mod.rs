//! Product quantization (Jégou, Douze, Schmid 2011).
//!
//! The vector is split into `m` contiguous subspaces; each gets its own
//! 256-centroid codebook trained with k-means (k-means++ seeding, Lloyd
//! iterations, rayon-parallel across subspaces). A vector compresses to
//! `m` bytes — for SIFT (128-dim f32, m=16) that is 32x.
//!
//! Search uses ADC (asymmetric distance computation): per query, one table
//! of `m x 256` exact subspace distances; the approximate distance to any
//! code is then `m` table lookups. Squared-L2 decomposes exactly over
//! subspaces, so ADC error comes only from quantization, not the metric.
//! L2 only in v1.

use rand::Rng;
use rayon::prelude::*;

use crate::distance::scalar::l2_squared;

pub const K: usize = 256; // centroids per subspace, one byte per code

#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    dim: usize,
    m: usize,
    sub_dim: usize,
    /// `m * K * sub_dim`, subspace-major.
    centroids: Vec<f32>,
}

impl ProductQuantizer {
    pub fn m(&self) -> usize {
        self.m
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    fn centroid(&self, sub: usize, k: usize) -> &[f32] {
        let off = (sub * K + k) * self.sub_dim;
        &self.centroids[off..off + self.sub_dim]
    }

    /// Train codebooks on `samples` (row-major, `dim` floats each).
    /// `dim` must be divisible by `m`.
    pub fn train<R: Rng>(
        samples: &[f32],
        dim: usize,
        m: usize,
        iters: usize,
        rng: &mut R,
    ) -> Self {
        assert!(dim % m == 0, "dim {dim} not divisible by m {m}");
        assert!(!samples.is_empty() && samples.len() % dim == 0);
        let n = samples.len() / dim;
        assert!(n >= K, "need at least {K} training vectors, got {n}");
        let sub_dim = dim / m;

        // Independent seed per subspace so rayon workers don't share rng.
        let seeds: Vec<u64> = (0..m).map(|_| rng.gen()).collect();

        let centroids: Vec<f32> = seeds
            .into_par_iter()
            .enumerate()
            .flat_map(|(sub, seed)| {
                let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
                // Gather this subspace's slice of every sample.
                let points: Vec<f32> = (0..n)
                    .flat_map(|i| {
                        let row = &samples[i * dim + sub * sub_dim..][..sub_dim];
                        row.iter().copied().collect::<Vec<_>>()
                    })
                    .collect();
                kmeans(&points, n, sub_dim, iters, &mut rng)
            })
            .collect();

        Self { dim, m, sub_dim, centroids }
    }

    /// Quantize `v` into `out` (`m` bytes).
    pub fn encode(&self, v: &[f32], out: &mut [u8]) {
        debug_assert_eq!(v.len(), self.dim);
        debug_assert_eq!(out.len(), self.m);
        for sub in 0..self.m {
            let x = &v[sub * self.sub_dim..][..self.sub_dim];
            let mut best = 0u8;
            let mut best_d = f32::INFINITY;
            for k in 0..K {
                let d = l2_squared(x, self.centroid(sub, k));
                if d < best_d {
                    best_d = d;
                    best = k as u8;
                }
            }
            out[sub] = best;
        }
    }

    /// Reconstruct the centroid approximation of a code (tests/debugging).
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.dim);
        for sub in 0..self.m {
            v.extend_from_slice(self.centroid(sub, code[sub] as usize));
        }
        v
    }

    /// Build the per-query ADC table: `m * K` exact subspace distances.
    pub fn adc_table(&self, query: &[f32]) -> Vec<f32> {
        debug_assert_eq!(query.len(), self.dim);
        let mut table = vec![0.0f32; self.m * K];
        for sub in 0..self.m {
            let q = &query[sub * self.sub_dim..][..self.sub_dim];
            let row = &mut table[sub * K..][..K];
            for k in 0..K {
                row[k] = l2_squared(q, self.centroid(sub, k));
            }
        }
        table
    }

    /// Serialize to bytes (checkpoint payload).
    pub fn to_bytes(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(3 + self.centroids.len());
        out.push(self.dim as f32);
        out.push(self.m as f32);
        out.push(self.sub_dim as f32);
        out.extend_from_slice(&self.centroids);
        out
    }

    pub fn from_bytes(data: &[f32]) -> Option<Self> {
        let (dim, m, sub_dim) = (data[0] as usize, data[1] as usize, data[2] as usize);
        let centroids = data[3..].to_vec();
        if dim == 0 || m == 0 || sub_dim * m != dim || centroids.len() != m * K * sub_dim {
            return None;
        }
        Some(Self { dim, m, sub_dim, centroids })
    }
}

/// ADC distance of one code against a query table. Four independent
/// accumulators so the loads pipeline; the table is 256 f32 per subspace
/// (1 KiB), so for realistic m the whole table lives in L1.
#[inline]
pub fn adc_distance(table: &[f32], code: &[u8]) -> f32 {
    let m = code.len();
    let mut acc = [0.0f32; 4];
    let mut sub = 0;
    while sub + 4 <= m {
        acc[0] += table[sub * K + code[sub] as usize];
        acc[1] += table[(sub + 1) * K + code[sub + 1] as usize];
        acc[2] += table[(sub + 2) * K + code[sub + 2] as usize];
        acc[3] += table[(sub + 3) * K + code[sub + 3] as usize];
        sub += 4;
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    while sub < m {
        sum += table[sub * K + code[sub] as usize];
        sub += 1;
    }
    sum
}

/// Lloyd's k-means with k-means++ seeding for one subspace.
/// `points` is `n` rows of `d` floats; returns `K * d` centroids.
fn kmeans<R: Rng>(points: &[f32], n: usize, d: usize, iters: usize, rng: &mut R) -> Vec<f32> {
    let row = |i: usize| &points[i * d..][..d];

    // k-means++ seeding: D^2-weighted sampling.
    let mut centroids = Vec::with_capacity(K * d);
    let first = rng.gen_range(0..n);
    centroids.extend_from_slice(row(first));
    let mut d2: Vec<f32> = (0..n).map(|i| l2_squared(row(i), row(first))).collect();
    for _ in 1..K {
        let total: f32 = d2.iter().sum();
        let pick = if total <= 0.0 {
            rng.gen_range(0..n) // all points identical to some centroid
        } else {
            let mut target = rng.gen_range(0.0..total);
            let mut chosen = n - 1;
            for (i, &w) in d2.iter().enumerate() {
                target -= w;
                if target <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        let start = centroids.len();
        centroids.extend_from_slice(row(pick));
        let new_c = &centroids[start..];
        for i in 0..n {
            let dd = l2_squared(row(i), new_c);
            if dd < d2[i] {
                d2[i] = dd;
            }
        }
    }

    // Lloyd iterations.
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        for i in 0..n {
            let mut best = 0;
            let mut best_d = f32::INFINITY;
            for k in 0..K {
                let dd = l2_squared(row(i), &centroids[k * d..][..d]);
                if dd < best_d {
                    best_d = dd;
                    best = k;
                }
            }
            assign[i] = best;
        }
        let mut sums = vec![0.0f64; K * d];
        let mut counts = vec![0u32; K];
        for i in 0..n {
            let k = assign[i];
            counts[k] += 1;
            for j in 0..d {
                sums[k * d + j] += points[i * d + j] as f64;
            }
        }
        for k in 0..K {
            if counts[k] == 0 {
                // Empty cluster: respawn on a random point.
                let p = row(rng.gen_range(0..n));
                centroids[k * d..][..d].copy_from_slice(p);
            } else {
                for j in 0..d {
                    centroids[k * d + j] = (sums[k * d + j] / counts[k] as f64) as f32;
                }
            }
        }
    }
    centroids
}

use rand::SeedableRng;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn blobs(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let centers: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..dim).map(|_| rng.gen_range(-3.0..3.0)).collect())
            .collect();
        (0..n)
            .flat_map(|_| {
                let c = &centers[rng.gen_range(0..centers.len())];
                c.iter().map(|x| x + rng.gen_range(-0.3..0.3)).collect::<Vec<f32>>()
            })
            .collect()
    }

    #[test]
    fn quantization_error_is_small_on_clustered_data() {
        let dim = 32;
        let data = blobs(4000, dim, 9);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let pq = ProductQuantizer::train(&data, dim, 8, 15, &mut rng);

        let mut code = vec![0u8; 8];
        let mut mse = 0.0f64;
        let mut var = 0.0f64;
        let n = data.len() / dim;
        let mean: Vec<f32> = {
            let mut acc = vec![0.0f32; dim];
            for i in 0..n {
                for j in 0..dim {
                    acc[j] += data[i * dim + j] / n as f32;
                }
            }
            acc
        };
        for i in 0..n {
            let v = &data[i * dim..][..dim];
            pq.encode(v, &mut code);
            mse += l2_squared(v, &pq.decode(&code)) as f64;
            var += l2_squared(v, &mean) as f64;
        }
        // Quantizer must explain almost all the variance of blobby data.
        assert!(mse / var < 0.05, "relative MSE {} too high", mse / var);
    }

    #[test]
    fn adc_matches_symmetric_distance_to_decoded() {
        // ADC(q, code) is exactly l2(q, decode(code)) by construction —
        // check the table + lookup plumbing agrees with the naive path.
        let dim = 16;
        let data = blobs(2000, dim, 4);
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let pq = ProductQuantizer::train(&data, dim, 4, 10, &mut rng);

        let mut code = vec![0u8; 4];
        for t in 0..50 {
            let q = &data[t * 31 * dim..][..dim];
            let v = &data[t * 17 * dim..][..dim];
            pq.encode(v, &mut code);
            let table = pq.adc_table(q);
            let adc = adc_distance(&table, &code);
            let direct = l2_squared(q, &pq.decode(&code));
            assert!(
                (adc - direct).abs() <= 1e-3 * direct.abs().max(1.0),
                "adc {adc} vs direct {direct}"
            );
        }
    }

    #[test]
    fn serialization_round_trip() {
        let dim = 16;
        let data = blobs(1000, dim, 8);
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let pq = ProductQuantizer::train(&data, dim, 4, 5, &mut rng);
        let restored = ProductQuantizer::from_bytes(&pq.to_bytes()).unwrap();
        let mut a = vec![0u8; 4];
        let mut b = vec![0u8; 4];
        for i in 0..20 {
            let v = &data[i * dim..][..dim];
            pq.encode(v, &mut a);
            restored.encode(v, &mut b);
            assert_eq!(a, b);
        }
    }
}
