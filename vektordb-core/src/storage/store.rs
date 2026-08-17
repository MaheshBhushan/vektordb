//! Memory-mapped vector store with lock-free reads and concurrent appends.
//!
//! File layout:
//!   [4 KiB header page][segment 0][segment 1]...
//! Segment k holds `BASE_ROWS << k` rows, so capacity doubles per segment and
//! ~40 segments cover any realistic size. Each segment is mmapped once, on
//! demand, into its own mapping that is **never moved or unmapped** while the
//! store is open — a `&[f32]` handed to a reader stays valid even while other
//! threads append and grow the file. This is what lets HNSW search run with
//! zero locks while inserts stream in.
//!
//! Rows are `dim * 4` bytes rounded up to 64: mappings are page-aligned and
//! segment byte offsets are multiples of the page size, so every row is
//! 64-byte aligned for the AVX2 kernels.
//!
//! Concurrency contract:
//! - `append` may be called from many threads; an internal mutex serializes
//!   appends (row copy included — appends are not the hot path, reads are).
//! - `count` is release-published after the row bytes are written; readers
//!   acquire-load it (or learn ids via graph links published with release
//!   semantics), so a visible id always reads fully-written data.
//!
//! Durability: writes land in the page cache via the shared mapping; the
//! header count is persisted by `flush()`. Crash consistency is the WAL's
//! job (M5) — it replays acked inserts on top of whatever was flushed.

use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use memmap2::{MmapMut, MmapOptions};
use parking_lot::Mutex;

use crate::error::{Error, Result};

const MAGIC: u32 = 0x564B_4442; // "VKDB"
const VERSION: u32 = 2;
const HEADER_SIZE: u64 = 4096;
/// Rows in segment 0. Multiple of 64 so that (rows * stride) is a multiple
/// of the page size and every segment offset is page-aligned.
const BASE_ROWS: u64 = 1024;
const MAX_SEGMENTS: usize = 40;
const ROW_ALIGN: usize = 64;

// Header field byte offsets.
const OFF_MAGIC: usize = 0;
const OFF_VERSION: usize = 4;
const OFF_DIM: usize = 8;
const OFF_COUNT: usize = 16;
const OFF_STRIDE: usize = 24;

/// Which segment holds row `id`, and the row's index within it.
/// Segment k spans rows `BASE_ROWS*(2^k - 1) .. BASE_ROWS*(2^(k+1) - 1)`.
#[inline]
fn locate(id: u64) -> (usize, u64) {
    let seg = (id / BASE_ROWS + 1).ilog2() as usize;
    let seg_start = BASE_ROWS * ((1u64 << seg) - 1);
    (seg, id - seg_start)
}

#[inline]
fn seg_rows(seg: usize) -> u64 {
    BASE_ROWS << seg
}

struct Appender {
    file: File,
    /// Segments `0..mapped` are mmapped.
    mapped: usize,
}

pub struct VectorStore {
    header: Mutex<MmapMut>,
    segments: [OnceLock<MmapMut>; MAX_SEGMENTS],
    appender: Mutex<Appender>,
    dim: usize,
    row_stride: usize,
    count: AtomicU64,
}

// Raw-pointer reads out of the mappings are what make this !Send by default;
// the concurrency contract above is exactly why sharing is sound.
unsafe impl Send for VectorStore {}
unsafe impl Sync for VectorStore {}

fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn read_u64(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

impl VectorStore {
    fn seg_byte_offset(&self, seg: usize) -> u64 {
        HEADER_SIZE + BASE_ROWS * ((1u64 << seg) - 1) * self.row_stride as u64
    }

    fn map_segment(file: &File, byte_offset: u64, rows: u64, stride: usize) -> Result<MmapMut> {
        let len = rows as usize * stride;
        let end = byte_offset + len as u64;
        if file.metadata()?.len() < end {
            file.set_len(end)?;
        }
        Ok(unsafe {
            MmapOptions::new()
                .offset(byte_offset)
                .len(len)
                .map_mut(file)?
        })
    }

    /// Create a new store file (truncating any existing one).
    pub fn create<P: AsRef<Path>>(path: P, dim: usize) -> Result<Self> {
        assert!(dim > 0, "dimension must be positive");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let row_stride = (dim * 4).div_ceil(ROW_ALIGN) * ROW_ALIGN;
        file.set_len(HEADER_SIZE)?;
        let mut header = unsafe {
            MmapOptions::new()
                .len(HEADER_SIZE as usize)
                .map_mut(&file)?
        };
        header[OFF_MAGIC..OFF_MAGIC + 4].copy_from_slice(&MAGIC.to_le_bytes());
        header[OFF_VERSION..OFF_VERSION + 4].copy_from_slice(&VERSION.to_le_bytes());
        header[OFF_DIM..OFF_DIM + 4].copy_from_slice(&(dim as u32).to_le_bytes());
        header[OFF_COUNT..OFF_COUNT + 8].copy_from_slice(&0u64.to_le_bytes());
        header[OFF_STRIDE..OFF_STRIDE + 8].copy_from_slice(&(row_stride as u64).to_le_bytes());
        header.flush()?;

        let store = Self {
            header: Mutex::new(header),
            segments: [const { OnceLock::new() }; MAX_SEGMENTS],
            appender: Mutex::new(Appender { file, mapped: 0 }),
            dim,
            row_stride,
            count: AtomicU64::new(0),
        };
        store.ensure_mapped_locked(&mut store.appender.lock(), 1)?;
        Ok(store)
    }

    /// Open an existing store.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        if file.metadata()?.len() < HEADER_SIZE {
            return Err(Error::Corrupt("file smaller than header".into()));
        }
        let header = unsafe {
            MmapOptions::new()
                .len(HEADER_SIZE as usize)
                .map_mut(&file)?
        };
        if read_u32(&header, OFF_MAGIC) != MAGIC {
            return Err(Error::Corrupt("bad magic".into()));
        }
        if read_u32(&header, OFF_VERSION) != VERSION {
            return Err(Error::Corrupt("unsupported version".into()));
        }
        let dim = read_u32(&header, OFF_DIM) as usize;
        let row_stride = read_u64(&header, OFF_STRIDE) as usize;
        let count = read_u64(&header, OFF_COUNT);
        if dim == 0 || row_stride < dim * 4 || !row_stride.is_multiple_of(ROW_ALIGN) {
            return Err(Error::Corrupt("inconsistent header geometry".into()));
        }

        let store = Self {
            header: Mutex::new(header),
            segments: [const { OnceLock::new() }; MAX_SEGMENTS],
            appender: Mutex::new(Appender { file, mapped: 0 }),
            dim,
            row_stride,
            count: AtomicU64::new(count),
        };
        // Map every segment that contains existing rows (plus segment 0).
        let (last_seg, _) = locate(count.saturating_sub(1));
        let needed = if count == 0 { 1 } else { last_seg + 1 };
        store.ensure_mapped_locked(&mut store.appender.lock(), needed)?;
        Ok(store)
    }

    fn ensure_mapped_locked(&self, app: &mut Appender, upto: usize) -> Result<()> {
        if upto > MAX_SEGMENTS {
            return Err(Error::Corrupt("store exceeds maximum size".into()));
        }
        while app.mapped < upto {
            let seg = app.mapped;
            let map = Self::map_segment(
                &app.file,
                self.seg_byte_offset(seg),
                seg_rows(seg),
                self.row_stride,
            )?;
            self.segments[seg]
                .set(map)
                .unwrap_or_else(|_| unreachable!("segment {seg} mapped twice"));
            app.mapped += 1;
        }
        Ok(())
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append a vector, returning its id. Callable from any thread.
    pub fn append(&self, vector: &[f32]) -> Result<u64> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        let mut app = self.appender.lock();
        let id = self.count.load(Ordering::Relaxed);
        let (seg, row) = locate(id);
        self.ensure_mapped_locked(&mut app, seg + 1)?;

        let map = self.segments[seg].get().unwrap();
        let off = row as usize * self.row_stride;
        // Sound despite &self: the append mutex makes this row ours alone,
        // and no reader may touch ids >= count yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                vector.as_ptr() as *const u8,
                map.as_ptr().add(off) as *mut u8,
                self.dim * 4,
            );
        }
        // Release-publish the new count after the row bytes are visible.
        self.count.store(id + 1, Ordering::Release);
        Ok(id)
    }

    /// Zero-copy read of a vector straight out of the mapping. The returned
    /// slice stays valid for the life of the store (segments never move).
    pub fn get(&self, id: u64) -> Result<&[f32]> {
        if id >= self.count.load(Ordering::Acquire) {
            return Err(Error::IdOutOfRange(id));
        }
        Ok(unsafe { self.get_unchecked(id) })
    }

    /// Like `get` but skips the bounds check; used by the index's hot loops
    /// where ids come from the graph and are known valid.
    ///
    /// # Safety
    /// `id` must be less than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, id: u64) -> &[f32] {
        let (seg, row) = locate(id);
        let map = self.segments.get_unchecked(seg).get().unwrap_unchecked();
        let off = row as usize * self.row_stride;
        std::slice::from_raw_parts(map.as_ptr().add(off) as *const f32, self.dim)
    }

    /// Persist the header count and msync all data pages.
    pub fn flush(&self) -> Result<()> {
        let app = self.appender.lock();
        let count = self.count.load(Ordering::Acquire);
        let mut header = self.header.lock();
        header[OFF_COUNT..OFF_COUNT + 8].copy_from_slice(&count.to_le_bytes());
        for seg in 0..app.mapped {
            self.segments[seg].get().unwrap().flush()?;
        }
        header.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_for(i: usize, dim: usize) -> Vec<f32> {
        (0..dim).map(|j| (i * dim + j) as f32 * 0.001).collect()
    }

    #[test]
    fn locate_maps_segments_correctly() {
        assert_eq!(locate(0), (0, 0));
        assert_eq!(locate(1023), (0, 1023));
        assert_eq!(locate(1024), (1, 0));
        assert_eq!(locate(3071), (1, 2047));
        assert_eq!(locate(3072), (2, 0));
        assert_eq!(locate(7168), (3, 0));
    }

    #[test]
    fn round_trip_and_alignment() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.store");
        let store = VectorStore::create(&path, 128).unwrap();
        for i in 0..100 {
            let id = store.append(&vec_for(i, 128)).unwrap();
            assert_eq!(id, i as u64);
        }
        for i in 0..100 {
            let v = store.get(i as u64).unwrap();
            assert_eq!(v, vec_for(i, 128).as_slice());
            assert_eq!(v.as_ptr() as usize % 64, 0, "row {i} not 64-byte aligned");
        }
        assert!(store.get(100).is_err());
    }

    #[test]
    fn grows_across_segments() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::create(dir.path().join("v.store"), 8).unwrap();
        // Hold a reference from segment 0 across growth into later segments:
        // segments never move, so it must stay valid and correct.
        store.append(&vec_for(0, 8)).unwrap();
        let early = store.get(0).unwrap();
        for i in 1..5000 {
            store.append(&vec_for(i, 8)).unwrap();
        }
        assert_eq!(store.len(), 5000);
        assert_eq!(early, vec_for(0, 8).as_slice());
        assert_eq!(store.get(4999).unwrap(), vec_for(4999, 8).as_slice());
    }

    #[test]
    fn reopen_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.store");
        {
            let store = VectorStore::create(&path, 32).unwrap();
            for i in 0..3000 {
                store.append(&vec_for(i, 32)).unwrap();
            }
            store.flush().unwrap();
        }
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(store.dim(), 32);
        assert_eq!(store.len(), 3000);
        for i in (0..3000).step_by(97) {
            assert_eq!(store.get(i as u64).unwrap(), vec_for(i, 32).as_slice());
        }
        // And appending after reopen lands in the right place.
        let id = store.append(&vec_for(3000, 32)).unwrap();
        assert_eq!(id, 3000);
        assert_eq!(store.get(3000).unwrap(), vec_for(3000, 32).as_slice());
    }

    #[test]
    fn concurrent_append_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let store =
            std::sync::Arc::new(VectorStore::create(dir.path().join("v.store"), 16).unwrap());
        let n_writers = 4;
        let per_writer = 2000;

        std::thread::scope(|s| {
            for _ in 0..n_writers {
                let store = store.clone();
                s.spawn(move || {
                    for i in 0..per_writer {
                        let v = vec![i as f32; 16];
                        let id = store.append(&v).unwrap();
                        // Own write must be immediately readable.
                        assert_eq!(store.get(id).unwrap(), v.as_slice());
                    }
                });
            }
            let store2 = store.clone();
            s.spawn(move || {
                // Reader hammers whatever is published; must never see junk
                // (every row is `[x; 16]` for some x — check self-consistency).
                for _ in 0..50_000 {
                    let n = store2.len();
                    if n == 0 {
                        continue;
                    }
                    let v = store2.get((n - 1) as u64).unwrap();
                    assert!(v.iter().all(|x| *x == v[0]), "torn read: {v:?}");
                }
            });
        });
        assert_eq!(store.len(), n_writers * per_writer);
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::create(dir.path().join("v.store"), 16).unwrap();
        assert!(store.append(&[0.0; 8]).is_err());
    }
}
