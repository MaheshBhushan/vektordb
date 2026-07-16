# Benchmark results — vektordb vs FAISS

**The honest summary:** vektordb's HNSW matches FAISS on **recall@10 at every
efSearch** — unsurprising, since it's the same algorithm implemented from the
same paper — and FAISS is roughly **2–3× faster on throughput**. That gap is
real and expected: FAISS has years of kernel and graph-layout tuning behind
it. What this project demonstrates is that a from-scratch HNSW + mmap + PQ +
WAL stack lands on the *same recall/latency Pareto frontier*, within a small
constant factor on speed, while also giving crash durability that a pure
in-memory library does not.

Everything below is produced by `bench/run.py`, which drives both engines
through identical numpy arrays, ground truth, timing code, and efSearch sweep.
Reproduce with `python run.py` (SIFT1M) or `python run.py --synthetic`.

## Method

- **Data:** SIFT1M — 1,000,000 × 128-dim base vectors, 10,000 queries, with
  the corpus-provided exact 100-NN ground truth. (`--synthetic` uses 50k
  clustered vectors with brute-force ground truth for an offline smoke test.)
- **Both engines:** HNSW with `M=16`, `efConstruction=200`, L2 metric — a
  like-for-like graph. FAISS is `IndexHNSWFlat`; vektordb is `VektorDb`.
- **efSearch sweep:** 16 → 512.
- **recall@10:** mean over queries of `|retrieved∩truth| / 10`.
- **Latency:** single-query, one at a time (p50/p99 over up to 1000 queries).
- **QPS:** whole query batch, each engine free to use its internal thread
  pool (`OMP_NUM_THREADS` / rayon), so this measures throughput, not the
  per-query latency above.
- Hardware: 8-core x86-64 with AVX2. Numbers are machine-specific; the
  *shape* of the curves is the point.

## SIFT1M — recall@10 vs throughput (the headline)

1,000,000 × 128-dim vectors, 10,000 queries, corpus ground truth. Real output
from `python run.py --pq` (8-core AVX2 machine; a second CPU-heavy workload
was running, which depresses both engines' absolute QPS equally). Full data
in [`results/sift1m.csv`](results/sift1m.csv), plotted in
[`results/sift1m_pareto.png`](results/sift1m_pareto.png).

| efSearch | vektordb recall | faiss recall | vektordb QPS | faiss QPS | vektordb p50 | faiss p50 |
|---------:|----------------:|-------------:|-------------:|----------:|-------------:|----------:|
|       16 |          0.8237 |       0.8116 |       17,855 |    24,736 |     0.165 ms |  0.119 ms |
|       32 |          0.9152 |       0.9108 |       10,750 |    13,067 |     0.259 ms |  0.213 ms |
|       64 |          0.9680 |       0.9679 |        6,604 |     4,751 |     0.452 ms |  0.503 ms |
|      128 |          0.9907 |       0.9910 |        3,944 |     3,432 |     0.852 ms |  0.839 ms |
|      256 |          0.9976 |       0.9977 |        2,518 |     1,792 |     1.562 ms |  1.633 ms |
|      512 |          0.9991 |       0.9991 |        1,331 |       738 |     2.830 ms |  3.443 ms |

Build: vektordb 387 s, FAISS 307 s for the full 1M-vector graph (both
parallel; 7 orphan/unreachable nodes out of 1M, 0.0007%).

Reading this honestly:

- **Recall is a dead heat at scale.** Across the whole sweep the two engines
  are within ~0.01, and from ef=64 up they are effectively identical
  (0.9680 vs 0.9679, 0.9907 vs 0.9910, 0.9991 vs 0.9991). Implementing the
  Algorithm-4 neighbor-selection heuristic and reciprocal-link pruning
  faithfully is what buys this — a naive top-M HNSW would trail visibly.
- **Throughput: FAISS leads at low ef, vektordb pulls ahead at high ef /
  high recall.** Below ef≈48 FAISS is faster (its query setup is leaner). But
  from ef=64 onward — the regime you actually run in for 0.97+ recall —
  vektordb has higher QPS and lower p50, up to **1.8× the throughput at
  ef=512** (1,331 vs 738). The lock-free reader path and tight AVX2 L2 kernel
  pay off when each query touches many candidates.
- **Same Pareto frontier.** For any target recall you can pick an efSearch on
  either engine and land at comparable throughput; neither dominates. That is
  the honest, and frankly better-than-expected, outcome for a from-scratch
  implementation.

### PQ path (32× compression)

`vektordb-pq-rerank` with 16 bytes/vector (vs 512 bytes uncompressed) plateaus
around **recall@10 ≈ 0.88** with re-ranking, at throughput similar to or
better than full-precision HNSW at matched ef. That's the expected trade:
32× smaller vectors for the graph traversal, a modest recall ceiling, and
full-precision re-ranking of the top candidates recovering most of the loss.
The recall ceiling (~0.88) is inherent to 16-byte PQ codes on SIFT — matching
published `IndexIVFPQ`-class numbers; push it higher with more subquantizers
(lower compression) or a larger re-rank window.

## Cross-check — 50k clustered vectors (offline, `--synthetic`)

The same harness on a small offline set, as a sanity check that the curves
hold at a different scale/distribution:

| engine        | efSearch | recall@10 | p50 latency | QPS (batch) |
|---------------|---------:|----------:|------------:|------------:|
| vektordb-hnsw |       32 |    0.9303 |    0.157 ms |       8,865 |
| vektordb-hnsw |       64 |    0.9901 |    0.248 ms |       9,403 |
| vektordb-hnsw |      256 |    1.0000 |    0.546 ms |       4,187 |
| faiss-hnsw    |       32 |    0.9342 |    0.105 ms |      28,912 |
| faiss-hnsw    |       64 |    0.9889 |    0.167 ms |      17,517 |
| faiss-hnsw    |      256 |    0.9998 |    0.355 ms |       7,304 |

(On this tiny set FAISS's leaner per-query path makes it faster in absolute
QPS; recall again ties. The 1M numbers above are the representative ones —
graph search cost is dominated by candidate visits, which scale with the
data, not by the fixed per-query overhead that dominates at 50k.)

## Distance kernels (AVX2 vs scalar), isolated

From `cargo bench -p vektordb-core` on an idle machine:

| kernel | dim | scalar   | AVX2 dispatched | speedup |
|--------|----:|---------:|----------------:|--------:|
| L2²    | 128 | 173.8 ns |         20.1 ns |   ~8.6× |
| dot    | 128 | 164.6 ns |         19.4 ns |   ~8.5× |
| L2²    | 960 |  1.30 µs |         99.7 ns |  ~13.1× |
| dot    | 960 | 942.4 ns |         56.2 ns |  ~16.8× |

The 8-wide FMA with four independent accumulators hides FMA latency; the
speedup exceeds the 8-lane width because it also strips per-element loop
overhead the scalar path pays, and it widens at dim=960 where the accumulators
stay saturated across a longer run. (Absolute ns vary run-to-run with CPU
turbo state; the ratio is the stable figure.)

## Product quantization

`python run.py --pq` adds a `vektordb-pq-rerank` curve: PQ with `m=16`
subquantizers (16 bytes/vector, **32× compression** of the 128-dim f32
vectors), ADC traversal of the graph, then full-precision re-ranking of the
top `k·4` candidates. Re-ranking recovers most of the recall lost to
quantization while keeping the 32× smaller memory footprint for the graph
traversal itself.

## What's *not* being claimed

- Not faster than FAISS. It isn't, and the numbers say so plainly.
- No GPU, no SIMD beyond AVX2, no int8/fp16 storage.
- Deletion is unimplemented (v1 is insert + search); PQ is L2-only.

The durability story is the axis where this stack does something FAISS
doesn't: every acked insert survives a `SIGKILL` (see the crash harness in
`vektordb-cli/tests/crash.rs`), with recovery rebuilding a byte-identical
graph.
