# vektordb

![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

A vector database built from first principles, in Rust — not a wrapper around
an existing ANN library. The point is to build the retrieval infrastructure
itself: the graph index, the on-disk storage, the compression, the distance
kernels, and the crash-recovery layer, then benchmark it honestly against
FAISS.

It brings together three areas in one artifact:

- **Algorithms** — HNSW implemented from the Malkov & Yashunin paper
  (hierarchical navigable small-world graphs), including the Algorithm-4
  neighbor-selection heuristic that actually matters for recall.
- **Systems** — memory-mapped storage with stable addresses, hand-written
  AVX2 distance kernels, lock-free concurrent reads with epoch-based
  reclamation, and a write-ahead log with crash recovery.
- **Databases** — durability (WAL + fsync + checkpoints), recovery semantics
  (torn-tail truncation, idempotent replay), and product-quantization
  compression for billion-scale memory budgets.

## Architecture

```
                    ┌──────────────────────────────────────────┐
   insert(vec) ───► │  Db  (db.rs)                              │
                    │   1. append to VectorStore  (mmap, M2)    │
                    │   2. append to WAL + fdatasync (M5)  ◄─ ack│
                    │   3. link into HNSW graph   (M3/M4)       │
                    └───────────┬──────────────────────────────┘
   search(q) ───────────────────┘  lock-free, no WAL, no store mutation
                    │
     ┌──────────────┼───────────────────────────────┐
     ▼              ▼               ▼                 ▼
 distance/      storage/         hnsw/              pq/
 scalar+AVX2    mmap store       graph + RCU        k-means codebooks
 (M1)           + exact k-NN     concurrency        + ADC search (M6)
                (M2)             (M3/M4)
```

On disk, a database is a directory:

```
mydb/
  vectors.store   fixed-stride, 64-byte-aligned mmap of raw f32 vectors
  index.snap      HNSW checkpoint (written temp+rename, CRC-checked)
  wal             append-only insert log since the last checkpoint
```

### Design decisions worth calling out

- **mmap segments never move.** The store grows by mapping new
  doubling-sized segments rather than remapping one region, so a `&[f32]`
  handed to a reader stays valid for the life of the store even while other
  threads append and the file grows. This is what lets search run completely
  lock-free.
- **RCU adjacency lists.** Each graph node's neighbor list is an immutable
  block behind a `crossbeam_epoch::Atomic`. Readers acquire-load the pointer;
  writers publish a replacement with a release store and retire the old block
  through the epoch collector. No reader ever takes a lock.
- **Deterministic level assignment.** HNSW level sampling is seeded from the
  vector id, not entropy, so recovery rebuilds a byte-identical graph and the
  whole index is a pure function of `(data, insert order)`. This turned a
  flaky crash test into a reproducible one — see below.
- **Durability vs. reachability are different contracts.** The WAL guarantees
  every acked vector's *bytes* survive any crash. Graph *reachability* is an
  approximate-index property: HNSW can, very rarely, leave a node with zero
  in-links ("unreachable point"), and this happens in clean, never-crashed
  builds too. The crash harness asserts durability strictly and treats the
  orphan rate as an aggregate health metric — it does not pretend crashes
  cause the unreachable-point property. (Verified: 20 clean builds of 12k
  vectors produced zero orphans; recovery of the same data produces the same
  graph.)

## Layout

| Crate            | What it is                                              |
|------------------|---------------------------------------------------------|
| `vektordb-core`  | the engine: distance, storage, hnsw, pq, wal, db        |
| `vektordb-cli`   | `ingest` / `verify` / `search` binary + crash harness   |
| `vektordb-py`    | PyO3 bindings (maturin), used by the benchmark           |
| `bench/`         | SIFT1M download, FAISS comparison, plots, RESULTS.md    |

## Build & test

```sh
cargo test --workspace --release       # unit + property + stress + crash tests
cargo bench -p vektordb-core           # AVX2 vs scalar distance kernels
```

Run the concurrency stress test under ThreadSanitizer (requires nightly):

```sh
RUSTFLAGS="-Zsanitizer=thread" \
  cargo +nightly test -Zbuild-std --target x86_64-unknown-linux-gnu \
  -p vektordb-core --release --lib -- concurrent
```

## Benchmark against FAISS

```sh
cd bench
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
( cd ../vektordb-py && maturin develop --release )   # build the extension

python run.py --synthetic     # offline, ~1 min
python run.py                 # real SIFT1M (downloads ~500 MB once)
python run.py --pq            # also sweep the PQ path
```

Both engines are driven through the same numpy arrays, ground truth, timing
code, and efSearch sweep. Full results and an honest write-up land in
[`bench/RESULTS.md`](bench/RESULTS.md).

**Headline (SIFT1M, 1M × 128-dim):** recall@10 ties FAISS across the whole
efSearch sweep (identical from ef≥64). On throughput neither engine
dominates — FAISS is faster below ef≈48, vektordb is faster in the high-recall
regime (ef≥64), reaching ~1.8× FAISS's QPS at ef=512 (0.9991 recall). Same
Pareto frontier, from a scratch implementation. The PQ path gives 32× vector
compression at ~0.88 recall@10 with re-ranking.

## Status / scope

v1 is insert + search. Deletion is deliberately out of scope — robust HNSW
deletion is a research topic of its own — and is noted as future work. PQ
currently supports the L2 metric only.
