//! vektordb-core: a small vector database engine built from first principles.
//!
//! Modules land milestone by milestone:
//! - `distance`: scalar + AVX2 kernels (M1)
//! - `storage`: mmap vector store + exact search (M2)
//! - `hnsw`: graph index (M3/M4)
//! - `wal`: write-ahead log (M5)
//! - `pq`: product quantization (M6)

pub mod db;
pub mod distance;
pub mod error;
pub mod hnsw;
pub mod pq;
pub mod storage;
pub mod wal;

pub use db::{Db, DbOptions};
pub use error::{Error, Result};
