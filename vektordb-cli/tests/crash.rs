//! Crash-recovery harness: spawn `vektordb ingest`, SIGKILL it at a random
//! moment, collect the ACKs it printed, and verify every acked insert
//! survives reopen. Repeated across randomized kill points, with occasional
//! checkpoints in the mix so recovery exercises snapshot + WAL-replay paths.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

const DIM: usize = 16;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_vektordb")
}

/// Run one ingest-kill-verify cycle. `floor` is the durable count proven by
/// earlier cycles; ingest resumes from the recovered length, which may
/// exceed it (an insert can be durable even if its ACK line never reached
/// us before the kill). Returns the new proven-durable count.
fn crash_cycle(
    dir: &std::path::Path,
    target: u64,
    kill_after_ms: u64,
    checkpoint_every: Option<u64>,
    floor: u64,
) -> u64 {
    let mut cmd = Command::new(bin());
    cmd.arg("ingest")
        .arg(dir)
        .arg(DIM.to_string())
        .arg(target.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    if let Some(every) = checkpoint_every {
        cmd.arg(every.to_string());
    }
    let mut child = cmd.spawn().expect("spawn ingest");

    // Drain ACK lines on a thread; the count when we kill is our floor.
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel::<String>();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(|l| l.ok()) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    std::thread::sleep(Duration::from_millis(kill_after_ms));
    let _ = child.kill(); // SIGKILL on unix: no destructors, no flushes
    let _ = child.wait();

    let mut acked = floor;
    let mut prev: Option<u64> = None;
    let mut done = false;
    while let Ok(line) = rx.try_recv() {
        if let Some(idstr) = line.strip_prefix("ACK ") {
            let id: u64 = idstr.trim().parse().expect("bad ACK line");
            match prev {
                None => assert!(id >= floor, "resumed below proven-durable floor"),
                Some(p) => assert_eq!(id, p + 1, "ACKs must be dense and ordered"),
            }
            prev = Some(id);
            acked = acked.max(id + 1);
        } else if line.starts_with("DONE") {
            done = true;
        }
    }
    reader.join().unwrap();
    if done {
        assert_eq!(acked, target);
    }

    // Recovery + verification in a fresh process, like a real restart.
    let out = Command::new(bin())
        .arg("verify")
        .arg(dir)
        .arg(DIM.to_string())
        .arg(acked.to_string())
        .output()
        .expect("spawn verify");
    assert!(
        out.status.success(),
        "verify failed after kill@{kill_after_ms}ms with {acked} acked:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    acked
}

#[test]
fn survives_sigkill_at_random_points() {
    // Fresh database, killed over and over at varying points. Later cycles
    // resume ingesting into the recovered database, so each iteration also
    // proves the recovered state accepts new writes.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("db");
    let mut total_acked = 0u64;
    let kill_points = [15, 40, 80, 150, 250, 400, 600, 900];
    for (i, &ms) in kill_points.iter().enumerate() {
        let checkpoint_every = if i % 3 == 2 { Some(500) } else { None };
        total_acked = crash_cycle(&dir, 1_000_000, ms, checkpoint_every, total_acked);
    }
    assert!(
        total_acked > 0,
        "no cycle acked anything — kill points too early"
    );
}

#[test]
fn clean_run_completes_and_verifies() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("db");
    let out = Command::new(bin())
        .arg("ingest")
        .arg(&dir)
        .arg(DIM.to_string())
        .arg("2000")
        .output()
        .unwrap();
    assert!(out.status.success());
    let out = Command::new(bin())
        .arg("verify")
        .arg(&dir)
        .arg(DIM.to_string())
        .arg("2000")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
