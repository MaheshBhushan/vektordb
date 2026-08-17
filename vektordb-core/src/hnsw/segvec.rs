//! Append-only segmented array with stable addresses and lock-free reads.
//!
//! Same doubling-segment scheme as the vector store: element addresses never
//! move, so readers can hold `&T` across concurrent growth. Writers claim
//! dense indices externally (HNSW node ids come from the store) and each
//! slot is written exactly once, before the id is published to readers via
//! a release store elsewhere (graph links / entry point).

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

use parking_lot::Mutex;

const BASE: usize = 1024;
const MAX_SEGMENTS: usize = 40;

#[inline]
fn locate(index: usize) -> (usize, usize) {
    let seg = (index / BASE + 1).ilog2() as usize;
    let seg_start = BASE * ((1usize << seg) - 1);
    (seg, index - seg_start)
}

#[inline]
fn seg_len(seg: usize) -> usize {
    BASE << seg
}

type Slot<T> = UnsafeCell<MaybeUninit<T>>;

pub struct SegVec<T> {
    segments: [AtomicPtr<Slot<T>>; MAX_SEGMENTS],
    /// Number of contiguously initialized slots (also the publish counter).
    len: AtomicU64,
    grow_lock: Mutex<()>,
}

unsafe impl<T: Send> Send for SegVec<T> {}
unsafe impl<T: Send + Sync> Sync for SegVec<T> {}

impl<T> SegVec<T> {
    pub fn new() -> Self {
        Self {
            segments: [const { AtomicPtr::new(std::ptr::null_mut()) }; MAX_SEGMENTS],
            len: AtomicU64::new(0),
            grow_lock: Mutex::new(()),
        }
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire) as usize
    }

    fn ensure_segment(&self, seg: usize) -> *mut Slot<T> {
        let p = self.segments[seg].load(Ordering::Acquire);
        if !p.is_null() {
            return p;
        }
        let _g = self.grow_lock.lock();
        let p = self.segments[seg].load(Ordering::Acquire);
        if !p.is_null() {
            return p;
        }
        let mut v: Vec<Slot<T>> = Vec::with_capacity(seg_len(seg));
        // MaybeUninit slots need no initialization; expose the full capacity.
        unsafe { v.set_len(seg_len(seg)) };
        let boxed = v.into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut Slot<T>;
        self.segments[seg].store(ptr, Ordering::Release);
        ptr
    }

    /// Write `value` into slot `index` and bump the publish counter.
    /// Indices must be claimed densely and each written exactly once; the
    /// caller (HNSW insert) guarantees this via the store's id allocation.
    pub fn set(&self, index: usize, value: T) {
        let (seg, off) = locate(index);
        let base = self.ensure_segment(seg);
        unsafe {
            (*base.add(off)).get().write(MaybeUninit::new(value));
        }
        // Publish: len only advances past `index` once the slot is written.
        // With dense one-shot writers, fetch_max keeps this monotone even if
        // ids finish out of order.
        self.len.fetch_max(index as u64 + 1, Ordering::Release);
    }

    /// Read slot `index`.
    ///
    /// # Safety
    /// The slot must have been initialized (`set` happened-before this call;
    /// in the index this is guaranteed because ids reach readers only
    /// through release-published links).
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        let (seg, off) = locate(index);
        let base = self.segments.get_unchecked(seg).load(Ordering::Acquire);
        (*(*base.add(off)).get()).assume_init_ref()
    }
}

impl<T> Drop for SegVec<T> {
    fn drop(&mut self) {
        let len = *self.len.get_mut() as usize;
        for seg in 0..MAX_SEGMENTS {
            let p = *self.segments[seg].get_mut();
            if p.is_null() {
                continue;
            }
            let seg_start = BASE * ((1usize << seg) - 1);
            let init = len.saturating_sub(seg_start).min(seg_len(seg));
            unsafe {
                for i in 0..init {
                    (*(*p.add(i)).get()).assume_init_drop();
                }
                drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                    p,
                    seg_len(seg),
                )));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_across_segments() {
        let v: SegVec<String> = SegVec::new();
        for i in 0..5000 {
            v.set(i, format!("item{i}"));
        }
        assert_eq!(v.len(), 5000);
        for i in (0..5000).step_by(111) {
            assert_eq!(unsafe { v.get_unchecked(i) }, &format!("item{i}"));
        }
        // Drop runs here; miri/asan builds would catch leaks or double-drops.
    }

    #[test]
    fn concurrent_dense_writers() {
        let v: std::sync::Arc<SegVec<u64>> = std::sync::Arc::new(SegVec::new());
        let next = std::sync::Arc::new(AtomicU64::new(0));
        std::thread::scope(|s| {
            for _ in 0..8 {
                let v = v.clone();
                let next = next.clone();
                s.spawn(move || loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= 20_000 {
                        break;
                    }
                    v.set(i as usize, i * 3);
                });
            }
        });
        assert_eq!(v.len(), 20_000);
        for i in 0..20_000u64 {
            assert_eq!(*unsafe { v.get_unchecked(i as usize) }, i * 3);
        }
    }
}
