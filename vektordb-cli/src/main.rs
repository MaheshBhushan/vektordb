//! vektordb CLI: ingest / verify / checkpoint demos, and the process the
//! crash-recovery harness SIGKILLs mid-ingest.
//!
//! Vectors are generated deterministically from their id (`vec_for_id`), so
//! `verify` can check byte-exact recovery without shipping data around.

use std::io::Write;

use vektordb_core::{Db, DbOptions};

fn vec_for_id(id: u64, dim: usize) -> Vec<f32> {
    // Cheap deterministic pseudo-random pattern, stable across runs.
    (0..dim as u64)
        .map(|j| {
            let x = id
                .wrapping_mul(6364136223846793005)
                .wrapping_add(j.wrapping_mul(1442695040888963407));
            ((x >> 33) as u32) as f32 / u32::MAX as f32 - 0.5
        })
        .collect()
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  vektordb ingest <dir> <dim> <count> [checkpoint_every]\n  vektordb verify <dir> <dim> <acked_count>\n  vektordb search <dir> <dim> <id> <k>"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
    }
    let cmd = args[1].as_str();
    let result = match cmd {
        "ingest" if args.len() >= 5 => ingest(
            &args[2],
            args[3].parse().unwrap_or_else(|_| usage()),
            args[4].parse().unwrap_or_else(|_| usage()),
            args.get(5).map(|s| s.parse().unwrap_or_else(|_| usage())),
        ),
        "verify" if args.len() == 5 => verify(
            &args[2],
            args[3].parse().unwrap_or_else(|_| usage()),
            args[4].parse().unwrap_or_else(|_| usage()),
        ),
        "orphans" if args.len() == 4 => orphans(
            &args[2],
            args[3].parse().unwrap_or_else(|_| usage()),
        ),
        "search" if args.len() == 6 => search(
            &args[2],
            args[3].parse().unwrap_or_else(|_| usage()),
            args[4].parse().unwrap_or_else(|_| usage()),
            args[5].parse().unwrap_or_else(|_| usage()),
        ),
        _ => usage(),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Insert `count` deterministic vectors, printing `ACK <id>` after each
/// durable insert. The harness kills this process at a random moment and
/// then verifies every acked id survived.
fn ingest(
    dir: &str,
    dim: usize,
    count: u64,
    checkpoint_every: Option<u64>,
) -> vektordb_core::Result<()> {
    let db = Db::open(dir, dim, DbOptions::default())?;
    let start = db.len() as u64;
    let stdout = std::io::stdout();
    for i in start..count {
        let id = db.insert(&vec_for_id(i, dim))?;
        // The ACK line is the durability promise; flush it immediately.
        let mut out = stdout.lock();
        writeln!(out, "ACK {id}").ok();
        out.flush().ok();
        if let Some(every) = checkpoint_every {
            if (i + 1) % every == 0 {
                db.checkpoint()?;
            }
        }
    }
    db.checkpoint()?;
    println!("DONE {count}");
    Ok(())
}

/// Reopen the database and verify recovery.
///
/// Two distinct guarantees, checked separately because they are different
/// contracts:
///
/// 1. Durability (the WAL's job — must be perfect): every acked insert's
///    bytes are recoverable and the count is at least `acked`. A single
///    mismatch here is a real data-loss bug.
///
/// 2. Graph health (an approximate-index property — must stay negligible):
///    the fraction of nodes with zero in-links ("unreachable points", a
///    documented HNSW property that also occurs in clean, never-crashed
///    builds). Recovery must not inflate this beyond what a clean build
///    produces, so we assert an aggregate rate, not per-node reachability.
///    Asserting the latter would test HNSW recall, not crash recovery.
fn verify(dir: &str, dim: usize, acked: u64) -> vektordb_core::Result<()> {
    let db = Db::open(dir, dim, DbOptions::default())?;

    // (1) Durability — strict.
    if (db.len() as u64) < acked {
        eprintln!("FAIL: db has {} vectors, {acked} were acked", db.len());
        std::process::exit(1);
    }
    for id in 0..acked {
        let expected = vec_for_id(id, dim);
        if db.get(id)? != expected.as_slice() {
            eprintln!("FAIL: vector {id} corrupt after recovery");
            std::process::exit(1);
        }
    }

    // (2) Graph health — aggregate.
    let orphans = db.orphan_count();
    let rate = orphans as f64 / (db.len().max(1)) as f64;
    if rate > 0.001 {
        eprintln!(
            "FAIL: {orphans}/{} nodes unreachable ({rate:.4}) — recovery degraded the graph",
            db.len()
        );
        std::process::exit(1);
    }
    println!("OK len={} orphans={orphans}", db.len());
    Ok(())
}

fn orphans(dir: &str, dim: usize) -> vektordb_core::Result<()> {
    let db = Db::open(dir, dim, DbOptions::default())?;
    println!("len={} orphans={}", db.len(), db.orphan_count());
    Ok(())
}

fn search(dir: &str, dim: usize, id: u64, k: usize) -> vektordb_core::Result<()> {
    let db = Db::open(dir, dim, DbOptions::default())?;
    for hit in db.search(&vec_for_id(id, dim), k, 128)? {
        println!("{}\t{}", hit.id, hit.distance);
    }
    Ok(())
}
