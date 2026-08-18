//! Distance kernels with one-time runtime dispatch.
//!
//! `Metric::kernel()` resolves to a plain `fn` pointer (AVX2 if the CPU has
//! it, scalar otherwise), so the hot search loops pay one indirect call and
//! no per-call feature detection.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

pub type DistanceFn = fn(&[f32], &[f32]) -> f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Squared L2. Monotonic in true L2, so it ranks identically without the
    /// square root.
    L2,
    /// Inner-product "distance": negated dot, so smaller is better for all
    /// metrics uniformly.
    Dot,
    /// Cosine *distance*: `1 - cos(a, b)`, so smaller is better here too.
    Cosine,
}

fn dot_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    -scalar::dot(a, b)
}

#[cfg(target_arch = "x86_64")]
fn l2_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe { avx2::l2_squared(a, b) }
}

#[cfg(target_arch = "x86_64")]
fn dot_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    -unsafe { avx2::dot(a, b) }
}

#[cfg(target_arch = "x86_64")]
fn cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe { avx2::cosine(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn have_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

impl Metric {
    /// Resolve this metric to the fastest available kernel. Cheap enough to
    /// call at index-open time; store the result.
    pub fn kernel(self) -> DistanceFn {
        #[cfg(target_arch = "x86_64")]
        if have_avx2() {
            return match self {
                Metric::L2 => l2_avx2,
                Metric::Dot => dot_distance_avx2,
                Metric::Cosine => cosine_avx2,
            };
        }
        match self {
            Metric::L2 => scalar::l2_squared,
            Metric::Dot => dot_distance_scalar,
            Metric::Cosine => scalar::cosine,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn vec_pair(max_len: usize) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
        (1..=max_len).prop_flat_map(|len| {
            let elem = prop_oneof![(-1000.0f32..1000.0), (-1.0f32..1.0), Just(0.0f32),];
            (
                prop::collection::vec(elem.clone(), len),
                prop::collection::vec(elem, len),
            )
        })
    }

    /// Relative tolerance: SIMD reduction order differs from scalar, so we
    /// compare against the magnitude of the accumulated terms.
    fn assert_close(simd: f32, reference: f32, scale: f32) {
        let tol = 1e-4f32 * scale.max(1.0);
        assert!(
            (simd - reference).abs() <= tol,
            "simd={simd} reference={reference} tol={tol}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    proptest! {
        #[test]
        fn avx2_l2_matches_scalar((a, b) in vec_pair(300)) {
            prop_assume!(have_avx2());
            let r = scalar::l2_squared(&a, &b);
            let s = unsafe { avx2::l2_squared(&a, &b) };
            assert_close(s, r, r.abs());
        }

        #[test]
        fn avx2_dot_matches_scalar((a, b) in vec_pair(300)) {
            prop_assume!(have_avx2());
            let scale: f32 = a.iter().zip(&b).map(|(x, y)| (x * y).abs()).sum();
            let r = scalar::dot(&a, &b);
            let s = unsafe { avx2::dot(&a, &b) };
            assert_close(s, r, scale);
        }

        #[test]
        fn avx2_cosine_matches_scalar((a, b) in vec_pair(300)) {
            prop_assume!(have_avx2());
            let r = scalar::cosine(&a, &b);
            let s = unsafe { avx2::cosine(&a, &b) };
            // Cosine is normalized, so an absolute tolerance is appropriate.
            prop_assert!((s - r).abs() < 1e-3, "simd={s} reference={r}");
        }
    }

    #[test]
    fn kernel_dispatch_works() {
        let a = vec![1.0f32; 128];
        let b = vec![2.0f32; 128];
        assert_eq!(Metric::L2.kernel()(&a, &b), 128.0);
        assert_eq!(Metric::Dot.kernel()(&a, &b), -256.0);
        let c = Metric::Cosine.kernel()(&a, &b);
        assert!(
            c.abs() < 1e-6,
            "parallel vectors should have cosine distance 0, got {c}"
        );
    }

    #[test]
    fn odd_lengths_hit_all_tail_paths() {
        // Exercise len in 1..=70 to cover the 32-, 8-, and 1-wide loops.
        for n in 1..=70usize {
            let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
            let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.25).collect();
            let k = Metric::L2.kernel();
            let r = scalar::l2_squared(&a, &b);
            assert_close(k(&a, &b), r, r.abs());
        }
    }
}
