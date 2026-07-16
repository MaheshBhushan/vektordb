"""SIFT1M loader.

Downloads the standard ANN benchmark set (Jégou et al., TEXMEX corpus):
1,000,000 base vectors x 128 dims, 10,000 queries, and precomputed
ground-truth 100-NN. Parsers for the .fvecs / .ivecs formats used by that
corpus. A tiny synthetic fallback keeps the harness runnable offline.
"""

import os
import tarfile
import urllib.request

import numpy as np

SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT_URL_HTTP = "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift.tar.gz"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _read_vecs(path, view_dtype):
    """Read a .fvecs/.ivecs file, vectorized.

    Each record is <int32 dim><dim values>. In the TEXMEX files every record
    has the same dim, so the file is a regular (n, dim+1) int32 grid: read it
    once, drop the leading dim column, and reinterpret the payload. This is
    O(1) numpy calls instead of a million-iteration Python loop.
    """
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    raw = raw.reshape(-1, dim + 1)
    if not np.all(raw[:, 0] == dim):
        raise ValueError(f"{path}: records have varying dimension")
    payload = np.ascontiguousarray(raw[:, 1:])
    return payload.view(view_dtype)


def read_fvecs(path):
    return _read_vecs(path, np.float32).astype(np.float32, copy=False)


def read_ivecs(path):
    return _read_vecs(path, np.int32).astype(np.int64)


def download_sift():
    os.makedirs(DATA_DIR, exist_ok=True)
    base_dir = os.path.join(DATA_DIR, "sift")
    if os.path.exists(os.path.join(base_dir, "sift_base.fvecs")):
        return base_dir
    tgz = os.path.join(DATA_DIR, "sift.tar.gz")
    if not os.path.exists(tgz):
        last = None
        for url in (SIFT_URL_HTTP, SIFT_URL):
            try:
                print(f"downloading SIFT1M from {url} ...")
                urllib.request.urlretrieve(url, tgz)
                last = None
                break
            except Exception as e:  # noqa: BLE001
                last = e
                print(f"  failed: {e}")
        if last is not None:
            raise RuntimeError(f"could not download SIFT1M: {last}")
    print("extracting ...")
    with tarfile.open(tgz) as t:
        t.extractall(DATA_DIR)
    return base_dir


def load_sift(limit=None):
    """Return (base, query, gt) as numpy arrays.

    base : (N, 128) float32   query : (Q, 128) float32
    gt   : (Q, 100) int64     ground-truth nearest-neighbor ids into base
    """
    base_dir = download_sift()
    base = read_fvecs(os.path.join(base_dir, "sift_base.fvecs"))
    query = read_fvecs(os.path.join(base_dir, "sift_query.fvecs"))
    gt = read_ivecs(os.path.join(base_dir, "sift_groundtruth.ivecs"))
    if limit is not None:
        base = base[:limit]
        # Ground truth is only valid against the full base; caller must
        # recompute it when subsetting (see synthetic()).
    return base, query, gt


def synthetic(n=50_000, dim=128, nq=1000, k=100, seed=0):
    """Offline fallback: clustered vectors with brute-force ground truth."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((50, dim)).astype(np.float32) * 5
    lbl = rng.integers(0, len(centers), n)
    base = (centers[lbl] + rng.standard_normal((n, dim)).astype(np.float32)).astype(np.float32)
    qlbl = rng.integers(0, len(centers), nq)
    query = (centers[qlbl] + rng.standard_normal((nq, dim)).astype(np.float32)).astype(np.float32)
    gt = brute_force_gt(base, query, k)
    return base, query, gt


def brute_force_gt(base, query, k):
    """Exact k-NN ground truth (L2), chunked to bound memory."""
    gt = np.empty((len(query), k), dtype=np.int64)
    base_sq = (base * base).sum(1)
    chunk = 256
    for i in range(0, len(query), chunk):
        q = query[i : i + chunk]
        # ||q-b||^2 = ||q||^2 + ||b||^2 - 2 q.b ; argpartition for top-k.
        d = base_sq[None, :] - 2.0 * q @ base.T
        idx = np.argpartition(d, k, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d, idx, axis=1), axis=1)
        gt[i : i + len(q)] = np.take_along_axis(idx, order, axis=1)
    return gt
