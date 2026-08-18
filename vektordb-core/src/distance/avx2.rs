//! AVX2+FMA kernels.
//!
//! Layout of each kernel: four independent 8-lane f32 accumulators walk the
//! input 32 floats per iteration so consecutive FMAs don't serialize on the
//! ~4-cycle FMA latency, then an 8-lane remainder loop, then a scalar tail
//! for `len % 8`. Reduction order therefore differs from the scalar kernels,
//! so results match only within floating-point tolerance, not bit-for-bit.
//!
//! Safety: every function here requires AVX2+FMA at runtime; the dispatcher
//! in `mod.rs` is the only caller and checks once at startup. Loads are
//! unaligned (`loadu`) so callers don't owe us alignment; the mmap store
//! aligns rows to 64 bytes anyway, which makes these loads full speed.

#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// Squared L2 distance between `a` and `b`.
///
/// # Safety
///
/// - AVX2 and FMA must be available on the running CPU. Use
///   [`Metric::kernel`](crate::distance::Metric::kernel), which feature-detects
///   once, rather than calling this directly.
/// - `a.len()` must equal `b.len()`. The lengths are only `debug_assert`ed, so
///   in release builds a shorter `b` reads out of bounds.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 32 <= n {
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
        let d1 = _mm256_sub_ps(
            _mm256_loadu_ps(pa.add(i + 8)),
            _mm256_loadu_ps(pb.add(i + 8)),
        );
        let d2 = _mm256_sub_ps(
            _mm256_loadu_ps(pa.add(i + 16)),
            _mm256_loadu_ps(pb.add(i + 16)),
        );
        let d3 = _mm256_sub_ps(
            _mm256_loadu_ps(pa.add(i + 24)),
            _mm256_loadu_ps(pb.add(i + 24)),
        );
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
        i += 32;
    }
    while i + 8 <= n {
        let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
        acc0 = _mm256_fmadd_ps(d, d, acc0);
        i += 8;
    }
    let mut sum = hsum256(_mm256_add_ps(
        _mm256_add_ps(acc0, acc1),
        _mm256_add_ps(acc2, acc3),
    ));
    while i < n {
        let d = *pa.add(i) - *pb.add(i);
        sum += d * d;
        i += 1;
    }
    sum
}

/// Dot product of `a` and `b`.
///
/// # Safety
///
/// Same contract as [`l2_squared`]: AVX2+FMA must be available on the running
/// CPU, and `a.len()` must equal `b.len()` (only `debug_assert`ed, so a
/// mismatch reads out of bounds in release builds).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 32 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 8)),
            _mm256_loadu_ps(pb.add(i + 8)),
            acc1,
        );
        acc2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 16)),
            _mm256_loadu_ps(pb.add(i + 16)),
            acc2,
        );
        acc3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 24)),
            _mm256_loadu_ps(pb.add(i + 24)),
            acc3,
        );
        i += 32;
    }
    while i + 8 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        i += 8;
    }
    let mut sum = hsum256(_mm256_add_ps(
        _mm256_add_ps(acc0, acc1),
        _mm256_add_ps(acc2, acc3),
    ));
    while i < n {
        sum += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    sum
}

/// Cosine *distance*: `1 - cos(a, b)`, computed in one pass (the three sums
/// accumulate together, so the vectors are read once). Zero-norm inputs yield
/// distance 1. Matches [`scalar::cosine`](crate::distance::scalar::cosine).
///
/// # Safety
///
/// Same contract as [`l2_squared`]: AVX2+FMA must be available on the running
/// CPU, and `a.len()` must equal `b.len()` (only `debug_assert`ed, so a
/// mismatch reads out of bounds in release builds).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());

    // One accumulator triple is enough here: three FMA chains per iteration
    // already give the FMA units independent work.
    let mut ab0 = _mm256_setzero_ps();
    let mut aa0 = _mm256_setzero_ps();
    let mut bb0 = _mm256_setzero_ps();
    let mut ab1 = _mm256_setzero_ps();
    let mut aa1 = _mm256_setzero_ps();
    let mut bb1 = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 16 <= n {
        let x0 = _mm256_loadu_ps(pa.add(i));
        let y0 = _mm256_loadu_ps(pb.add(i));
        let x1 = _mm256_loadu_ps(pa.add(i + 8));
        let y1 = _mm256_loadu_ps(pb.add(i + 8));
        ab0 = _mm256_fmadd_ps(x0, y0, ab0);
        aa0 = _mm256_fmadd_ps(x0, x0, aa0);
        bb0 = _mm256_fmadd_ps(y0, y0, bb0);
        ab1 = _mm256_fmadd_ps(x1, y1, ab1);
        aa1 = _mm256_fmadd_ps(x1, x1, aa1);
        bb1 = _mm256_fmadd_ps(y1, y1, bb1);
        i += 16;
    }
    while i + 8 <= n {
        let x = _mm256_loadu_ps(pa.add(i));
        let y = _mm256_loadu_ps(pb.add(i));
        ab0 = _mm256_fmadd_ps(x, y, ab0);
        aa0 = _mm256_fmadd_ps(x, x, aa0);
        bb0 = _mm256_fmadd_ps(y, y, bb0);
        i += 8;
    }
    let mut ab = hsum256(_mm256_add_ps(ab0, ab1));
    let mut aa = hsum256(_mm256_add_ps(aa0, aa1));
    let mut bb = hsum256(_mm256_add_ps(bb0, bb1));
    while i < n {
        let (x, y) = (*pa.add(i), *pb.add(i));
        ab += x * y;
        aa += x * x;
        bb += y * y;
        i += 1;
    }
    let denom = (aa * bb).sqrt();
    if denom == 0.0 {
        1.0
    } else {
        1.0 - ab / denom
    }
}
