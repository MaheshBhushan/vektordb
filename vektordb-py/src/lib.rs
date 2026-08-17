//! Python bindings for vektordb, built with maturin/PyO3.
//!
//! One class, `VektorDb`, wrapping the core `Db`. numpy arrays cross the
//! boundary zero-copy where possible; the heavy calls (`add`, `search`,
//! `train_pq`) release the GIL so a Python benchmark harness can drive this
//! and FAISS through identical code paths and thread pools.
//!
//! `useless_conversion` is allowed crate-wide because pyo3 0.22's
//! `#[pymethods]` expansion converts every `PyResult` error with an `.into()`
//! that is a no-op when the error is already a `PyErr`. Clippy blames our
//! return types, where there is nothing to remove, and macro hygiene means
//! neither an impl-level nor a per-method allow suppresses it (both tried).
//! Revisit when pyo3 is upgraded.
#![allow(clippy::useless_conversion)]

use numpy::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use vektordb_core::distance::Metric;
use vektordb_core::hnsw::HnswConfig;
use vektordb_core::wal::SyncPolicy;
use vektordb_core::{Db, DbOptions};

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// `(ids, distances)`, both shape `(nq, k)` — what every search returns.
type Neighbors<'py> = (Bound<'py, PyArray2<u64>>, Bound<'py, PyArray2<f32>>);

/// A vektordb index backed by a directory on disk.
///
/// Parameters
/// ----------
/// path : str        directory for the mmap store / snapshot / WAL
/// dim : int         vector dimensionality
/// m, ef_construction : int   HNSW build parameters
/// metric : "l2" | "ip" | "cosine"
/// durable : bool    enable the write-ahead log (default False, for speed)
#[pyclass]
struct VektorDb {
    db: Db,
    dim: usize,
}

#[pymethods]
impl VektorDb {
    #[new]
    #[pyo3(signature = (path, dim, m=16, ef_construction=200, metric="l2", durable=false))]
    fn new(
        path: &str,
        dim: usize,
        m: usize,
        ef_construction: usize,
        metric: &str,
        durable: bool,
    ) -> PyResult<Self> {
        let metric = match metric {
            "l2" => Metric::L2,
            "ip" | "dot" => Metric::Dot,
            "cosine" => Metric::Cosine,
            other => return Err(PyValueError::new_err(format!("unknown metric {other:?}"))),
        };
        let opts = DbOptions {
            config: HnswConfig {
                m,
                ef_construction,
                metric,
            },
            sync: SyncPolicy::Always,
            enable_wal: durable,
        };
        let db = Db::open(path, dim, opts).map_err(err)?;
        Ok(Self { db, dim })
    }

    /// Add a batch of vectors, shape (n, dim), float32. Returns the id of
    /// the first inserted vector. Releases the GIL for the parallel build.
    fn add(&self, py: Python<'_>, vectors: PyReadonlyArray2<'_, f32>) -> PyResult<u64> {
        if vectors.shape()[1] != self.dim {
            return Err(PyValueError::new_err(format!(
                "expected dim {}, got {}",
                self.dim,
                vectors.shape()[1]
            )));
        }
        let flat = vectors.as_slice()?; // requires C-contiguous
        let owned = flat.to_vec();
        py.allow_threads(|| self.db.add_batch(&owned)).map_err(err)
    }

    /// Search `queries` (shape (nq, dim), float32) for the `k` nearest
    /// neighbors of each. Returns (ids, distances), each shape (nq, k).
    /// Rows shorter than k (tiny indexes) are padded with id u64::MAX and
    /// +inf. Releases the GIL and parallelizes across queries.
    #[pyo3(signature = (queries, k, ef=64))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f32>,
        k: usize,
        ef: usize,
    ) -> PyResult<Neighbors<'py>> {
        if queries.shape()[1] != self.dim {
            return Err(PyValueError::new_err("query dim mismatch"));
        }
        let nq = queries.shape()[0];
        let flat = queries.as_slice()?.to_vec();
        let dim = self.dim;

        let (ids, dists) = py.allow_threads(|| {
            use rayon::prelude::*;
            let mut ids = vec![u64::MAX; nq * k];
            let mut dists = vec![f32::INFINITY; nq * k];
            ids.par_chunks_mut(k)
                .zip(dists.par_chunks_mut(k))
                .enumerate()
                .for_each(|(q, (id_row, dist_row))| {
                    let query = &flat[q * dim..][..dim];
                    let hits = self.db.search(query, k, ef).unwrap_or_default();
                    for (j, h) in hits.iter().enumerate() {
                        id_row[j] = h.id;
                        dist_row[j] = h.distance;
                    }
                });
            (ids, dists)
        });

        Ok((rows_u64(py, ids, nq, k)?, rows_f32(py, dists, nq, k)?))
    }

    /// Search over PQ codes with ADC + optional full-precision re-ranking.
    /// `train_pq` must have been called first.
    #[pyo3(signature = (queries, k, ef=64, rerank=4))]
    fn search_pq<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f32>,
        k: usize,
        ef: usize,
        rerank: usize,
    ) -> PyResult<Neighbors<'py>> {
        if queries.shape()[1] != self.dim {
            return Err(PyValueError::new_err("query dim mismatch"));
        }
        let nq = queries.shape()[0];
        let flat = queries.as_slice()?.to_vec();
        let dim = self.dim;

        let (ids, dists) = py
            .allow_threads(|| {
                use rayon::prelude::*;
                let mut ids = vec![u64::MAX; nq * k];
                let mut dists = vec![f32::INFINITY; nq * k];
                ids.par_chunks_mut(k)
                    .zip(dists.par_chunks_mut(k))
                    .enumerate()
                    .try_for_each(|(q, (id_row, dist_row))| {
                        let query = &flat[q * dim..][..dim];
                        let hits = self.db.search_pq(query, k, ef, rerank)?;
                        for (j, h) in hits.iter().enumerate() {
                            id_row[j] = h.id;
                            dist_row[j] = h.distance;
                        }
                        Ok::<_, vektordb_core::Error>(())
                    })
                    .map(|_| (ids, dists))
            })
            .map_err(err)?;

        Ok((rows_u64(py, ids, nq, k)?, rows_f32(py, dists, nq, k)?))
    }

    /// Train product quantization on the current contents.
    #[pyo3(signature = (m=16, iters=25, max_samples=100_000))]
    fn train_pq(&self, py: Python<'_>, m: usize, iters: usize, max_samples: usize) -> PyResult<()> {
        py.allow_threads(|| self.db.train_pq(m, iters, max_samples))
            .map_err(err)
    }

    /// Persist a snapshot and flush the store.
    fn checkpoint(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.db.checkpoint()).map_err(err)
    }

    /// Reconstruct a single stored vector (copy) as a numpy array.
    fn get<'py>(&self, py: Python<'py>, id: u64) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let v = self.db.get(id).map_err(err)?.to_vec();
        Ok(PyArray1::from_slice_bound(py, &v))
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    fn __len__(&self) -> usize {
        self.db.len()
    }

    /// Number of unreachable (zero-in-link) nodes — graph health metric.
    fn orphan_count(&self) -> usize {
        self.db.orphan_count()
    }
}

/// Wrap a flat `n*k` buffer as an (n, k) numpy array (one copy in, then a
/// zero-copy reshape).
fn rows_u64(
    py: Python<'_>,
    flat: Vec<u64>,
    n: usize,
    k: usize,
) -> PyResult<Bound<'_, PyArray2<u64>>> {
    PyArray1::from_slice_bound(py, &flat).reshape([n, k])
}

fn rows_f32(
    py: Python<'_>,
    flat: Vec<f32>,
    n: usize,
    k: usize,
) -> PyResult<Bound<'_, PyArray2<f32>>> {
    PyArray1::from_slice_bound(py, &flat).reshape([n, k])
}

#[pymodule]
fn vektordb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VektorDb>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
