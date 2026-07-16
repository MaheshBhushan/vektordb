"""Honest recall@10-vs-QPS benchmark: vektordb vs FAISS on SIFT1M.

Both engines see the same numpy arrays, the same ground truth, the same
timing code, and the same efSearch sweep. We report where FAISS wins and
where we're close; nothing is cherry-picked. Output: a CSV and a Pareto
plot under bench/results/, plus a printed table.

Usage:
    python run.py                 # SIFT1M (downloads ~500MB once)
    python run.py --synthetic     # offline clustered fallback
    python run.py --limit 200000  # subset of SIFT base (recomputes GT)
"""

import argparse
import csv
import os
import sys
import tempfile
import time

import numpy as np

import datasets

# A build+sweep over 1M vectors takes minutes; keep progress visible even
# when stdout is a pipe (e.g. `| tee`).
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# HNSW build params shared by both engines for a like-for-like graph.
M = 16
EF_CONSTRUCTION = 200
EF_SWEEP = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
K = 10


def recall_at_k(got_ids, gt_ids, k):
    """Fraction of true k-NN retrieved, averaged over queries."""
    hits = 0
    for got, truth in zip(got_ids, gt_ids):
        hits += len(set(got[:k]).intersection(truth[:k]))
    return hits / (len(got_ids) * k)


def time_queries(fn, query, warmup=1):
    """Return (per-query latencies ms, wall seconds) for fn(query)->ids."""
    for _ in range(warmup):
        fn(query[: min(64, len(query))])
    # Single-threaded latency distribution: one query at a time.
    lat = np.empty(min(1000, len(query)))
    for i in range(len(lat)):
        t0 = time.perf_counter()
        fn(query[i : i + 1])
        lat[i] = (time.perf_counter() - t0) * 1e3
    # Throughput: whole batch (engine may parallelize internally).
    t0 = time.perf_counter()
    ids = fn(query)
    wall = time.perf_counter() - t0
    return lat, wall, ids


def bench_vektordb(base, query, gt, rows):
    import vektordb

    dim = base.shape[1]
    with tempfile.TemporaryDirectory() as td:
        db = vektordb.VektorDb(td, dim, m=M, ef_construction=EF_CONSTRUCTION, metric="l2")
        t0 = time.perf_counter()
        db.add(np.ascontiguousarray(base))
        build = time.perf_counter() - t0
        print(f"[vektordb] build {len(db)} vectors in {build:.1f}s  orphans={db.orphan_count()}")

        for ef in EF_SWEEP:
            fn = lambda q, ef=ef: db.search(np.ascontiguousarray(q), K, ef=ef)[0]
            lat, wall, ids = time_queries(fn, query)
            rec = recall_at_k(ids, gt, K)
            rows.append(_row("vektordb-hnsw", ef, rec, lat, wall, len(query), build))
            print(f"  ef={ef:4d}  recall@10={rec:.4f}  p50={np.percentile(lat,50):.3f}ms  QPS={len(query)/wall:,.0f}")


def bench_vektordb_pq(base, query, gt, rows):
    import vektordb

    dim = base.shape[1]
    with tempfile.TemporaryDirectory() as td:
        db = vektordb.VektorDb(td, dim, m=M, ef_construction=EF_CONSTRUCTION, metric="l2")
        db.add(np.ascontiguousarray(base))
        t0 = time.perf_counter()
        db.train_pq(m=16, iters=25)
        train = time.perf_counter() - t0
        print(f"[vektordb-pq] trained PQ (16 bytes/vec, {dim*4//16}x compression) in {train:.1f}s")
        for ef in EF_SWEEP:
            fn = lambda q, ef=ef: db.search_pq(np.ascontiguousarray(q), K, ef=ef, rerank=4)[0]
            lat, wall, ids = time_queries(fn, query)
            rec = recall_at_k(ids, gt, K)
            rows.append(_row("vektordb-pq-rerank", ef, rec, lat, wall, len(query), train))
            print(f"  ef={ef:4d}  recall@10={rec:.4f}  p50={np.percentile(lat,50):.3f}ms  QPS={len(query)/wall:,.0f}")


def bench_faiss(base, query, gt, rows):
    try:
        import faiss
    except ImportError:
        print("faiss not installed; skipping")
        return

    dim = base.shape[1]
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    t0 = time.perf_counter()
    index.add(np.ascontiguousarray(base))
    build = time.perf_counter() - t0
    print(f"[faiss] build {index.ntotal} vectors in {build:.1f}s")

    for ef in EF_SWEEP:
        index.hnsw.efSearch = ef
        fn = lambda q: index.search(np.ascontiguousarray(q), K)[1]
        lat, wall, ids = time_queries(fn, query)
        rec = recall_at_k(ids, gt, K)
        rows.append(_row("faiss-hnsw", ef, rec, lat, wall, len(query), build))
        print(f"  ef={ef:4d}  recall@10={rec:.4f}  p50={np.percentile(lat,50):.3f}ms  QPS={len(query)/wall:,.0f}")


def _row(engine, ef, recall, lat, wall, nq, build_s):
    return {
        "engine": engine,
        "ef_search": ef,
        "recall@10": round(recall, 5),
        "p50_ms": round(float(np.percentile(lat, 50)), 4),
        "p99_ms": round(float(np.percentile(lat, 99)), 4),
        "qps": round(nq / wall, 1),
        "build_or_train_s": round(build_s, 2),
    }


def plot(rows, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    plt.figure(figsize=(8, 6))
    for engine in sorted({r["engine"] for r in rows}):
        pts = sorted((r["recall@10"], r["qps"]) for r in rows if r["engine"] == engine)
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker="o", label=engine)
    plt.xlabel("recall@10")
    plt.ylabel("queries/sec (batch, higher is better)")
    plt.yscale("log")
    plt.title("SIFT1M: recall@10 vs throughput (efSearch sweep)")
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="offline clustered data")
    ap.add_argument("--limit", type=int, default=None, help="subset of SIFT base")
    ap.add_argument("--pq", action="store_true", help="also benchmark the PQ path")
    args = ap.parse_args()

    if args.synthetic:
        print("using synthetic clustered dataset")
        base, query, gt = datasets.synthetic()
    elif args.limit:
        base, query, _ = datasets.load_sift()
        base = np.ascontiguousarray(base[: args.limit])
        print(f"subset {len(base)} base vectors; recomputing ground truth ...")
        gt = datasets.brute_force_gt(base, query, 100)
    else:
        base, query, gt = datasets.load_sift()
    print(f"base={base.shape} query={query.shape} gt={gt.shape}")

    rows = []
    bench_vektordb(base, query, gt, rows)
    bench_faiss(base, query, gt, rows)
    if args.pq:
        bench_vektordb_pq(base, query, gt, rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "sift1m.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")
    plot(rows, os.path.join(RESULTS_DIR, "sift1m_pareto.png"))


if __name__ == "__main__":
    main()
