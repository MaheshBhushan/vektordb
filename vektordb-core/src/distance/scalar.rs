//! Portable reference kernels. These define the semantics that the SIMD
//! kernels must match; every SIMD implementation is property-tested against
//! these.

#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "distance vectors must have equal lengths");
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        acc += d * d;
    }
    acc
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "distance vectors must have equal lengths");
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

/// Cosine *distance*: 1 - cos(a, b). Zero-norm inputs yield distance 1.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "distance vectors must have equal lengths");
    let (mut ab, mut aa, mut bb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    let denom = (aa * bb).sqrt();
    if denom == 0.0 {
        1.0
    } else {
        1.0 - ab / denom
    }
}
