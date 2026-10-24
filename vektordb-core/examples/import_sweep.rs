//! Bounded-batch import sweep: loads the same random dataset through
//! `Db::add_batch` at several batch sizes, then drops and reopens the
//! database to measure WAL recovery and verify every record came back.
//!
//! ```text
//! cargo run --release -p vektordb-core --example import_sweep -- \
//!     [--n 20000] [--dim 64] [--batches 1,64,1024,8192] [--sync never|always] \
//!     [--checkpoint] [--seed 7]
//! ```
//!
//! Prints one CSV row per batch size to stdout. Exits non-zero if any
//! batch size recovers a different record count or a mismatching vector.
//! Memory columns come from `/proc/self/status` (Linux); they read 0 elsewhere.

use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Instant;
use vektordb_core::{wal::SyncPolicy, Db, DbOptions};

struct Args {
    n: usize,
    dim: usize,
    batches: Vec<usize>,
    sync: SyncPolicy,
    checkpoint: bool,
    seed: u64,
}

fn parse_args() -> Args {
    let mut args = Args {
        n: 20_000,
        dim: 64,
        batches: vec![1, 64, 1024, 8192],
        sync: SyncPolicy::Never,
        checkpoint: false,
        seed: 7,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| usage(&format!("{flag} needs a value")));
        match flag.as_str() {
            "--n" => args.n = value().parse().unwrap_or_else(|_| usage("bad --n")),
            "--dim" => args.dim = value().parse().unwrap_or_else(|_| usage("bad --dim")),
            "--seed" => args.seed = value().parse().unwrap_or_else(|_| usage("bad --seed")),
            "--batches" => {
                args.batches = value()
                    .split(',')
                    .map(|s| match s.trim().parse() {
                        Ok(b) if b > 0 => b,
                        _ => usage("bad --batches"),
                    })
                    .collect()
            }
            "--sync" => {
                args.sync = match value().as_str() {
                    "always" => SyncPolicy::Always,
                    "never" => SyncPolicy::Never,
                    _ => usage("--sync must be always or never"),
                }
            }
            "--checkpoint" => args.checkpoint = true,
            "-h" | "--help" => usage(""),
            other => usage(&format!("unknown flag {other}")),
        }
    }
    if args.n == 0 || args.dim == 0 {
        usage("--n and --dim must be positive");
    }
    args
}

fn usage(msg: &str) -> ! {
    if !msg.is_empty() {
        eprintln!("error: {msg}");
    }
    eprintln!(
        "usage: import_sweep [--n N] [--dim D] [--batches B1,B2,..] \
         [--sync never|always] [--checkpoint] [--seed S]"
    );
    std::process::exit(if msg.is_empty() { 0 } else { 2 });
}

/// Value in KiB of a `/proc/self/status` field such as `VmRSS`.
fn proc_kib(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(field) && l[field.len()..].starts_with(':'))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

/// Reset the peak-RSS watermark so each batch size reports its own peak.
fn reset_peak_rss() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

fn main() {
    let args = parse_args();
    let mut rng = SmallRng::seed_from_u64(args.seed);
    let data: Vec<f32> = (0..args.n * args.dim).map(|_| rng.gen::<f32>()).collect();
    let sync_name = match args.sync {
        SyncPolicy::Always => "always",
        SyncPolicy::Never => "never",
    };

    println!(
        "batch_size,n,dim,sync,checkpoint,import_s,vectors_per_s,rss_before_kib,\
         rss_after_kib,peak_rss_kib,recover_s,recovered,mismatched,ok"
    );
    let mut failures = 0;
    for &batch in &args.batches {
        let dir = tempfile::tempdir().expect("tempdir");
        let opts = || DbOptions {
            sync: args.sync,
            ..DbOptions::default()
        };

        reset_peak_rss();
        let rss_before = proc_kib("VmRSS");
        let start = Instant::now();
        {
            let db = Db::open(dir.path(), args.dim, opts()).expect("open");
            for chunk in data.chunks(batch * args.dim) {
                db.add_batch(chunk).expect("add_batch");
            }
            if args.checkpoint {
                db.checkpoint().expect("checkpoint");
            }
        }
        let import_s = start.elapsed().as_secs_f64();
        let rss_after = proc_kib("VmRSS");
        let peak_rss = proc_kib("VmHWM");

        // "Restart": everything above was dropped; reopen from disk only.
        let start = Instant::now();
        let db = Db::open(dir.path(), args.dim, opts()).expect("reopen");
        let recover_s = start.elapsed().as_secs_f64();
        let recovered = db.len();
        let mismatched = (0..recovered.min(args.n))
            .filter(|&i| {
                db.get(i as u64).ok() != Some(&data[i * args.dim..(i + 1) * args.dim])
            })
            .count();
        let ok = recovered == args.n && mismatched == 0;
        failures += usize::from(!ok);

        println!(
            "{batch},{},{},{sync_name},{},{import_s:.4},{:.0},{rss_before},{rss_after},\
             {peak_rss},{recover_s:.4},{recovered},{mismatched},{ok}",
            args.n,
            args.dim,
            args.checkpoint,
            args.n as f64 / import_s,
        );
    }
    if failures > 0 {
        eprintln!("{failures} batch size(s) failed recovery verification");
        std::process::exit(1);
    }
}
