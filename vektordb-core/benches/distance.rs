use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use vektordb_core::distance::{scalar, Metric};

fn make_vec(n: usize, seed: u32) -> Vec<f32> {
    // Deterministic pseudo-random floats without pulling rand into the bench.
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn bench_kernels(c: &mut Criterion) {
    for &dim in &[128usize, 960] {
        let a = make_vec(dim, 1);
        let b = make_vec(dim, 2);
        let bytes = (2 * dim * std::mem::size_of::<f32>()) as u64;

        let mut group = c.benchmark_group(format!("l2_dim{dim}"));
        group.throughput(Throughput::Bytes(bytes));
        group.bench_function(BenchmarkId::new("scalar", dim), |bch| {
            bch.iter(|| scalar::l2_squared(black_box(&a), black_box(&b)))
        });
        let kernel = Metric::L2.kernel();
        group.bench_function(BenchmarkId::new("dispatched", dim), |bch| {
            bch.iter(|| kernel(black_box(&a), black_box(&b)))
        });
        group.finish();

        let mut group = c.benchmark_group(format!("dot_dim{dim}"));
        group.throughput(Throughput::Bytes(bytes));
        group.bench_function(BenchmarkId::new("scalar", dim), |bch| {
            bch.iter(|| scalar::dot(black_box(&a), black_box(&b)))
        });
        let kernel = Metric::Dot.kernel();
        group.bench_function(BenchmarkId::new("dispatched", dim), |bch| {
            bch.iter(|| kernel(black_box(&a), black_box(&b)))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_kernels);
criterion_main!(benches);
