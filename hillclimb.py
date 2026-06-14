#!/usr/bin/env python3
"""
Hill-climbing parameter search for seam carving scoring.
Resumable: reads hillclimb_log.jsonl to skip already-evaluated configs.
Uses a lock file to prevent duplicate instances.

Parameters (env vars fed to the binary):
  SEAM_INK_POWER, SEAM_INK_NORM, SEAM_INK_ROW_WEIGHT, SEAM_INK_ROW_POWER,
  SEAM_DELTA_WEIGHT, SEAM_DELTA_POWER, SEAM_DELTA_SCALE_POWER,
  SEAM_DELTA_ROW_WEIGHT, SEAM_DELTA_ROW_POWER
"""

import subprocess, re, os, sys, json, time, fcntl, hashlib, argparse
from pathlib import Path

CARGO = os.path.expanduser("~/.cargo/bin/cargo")
REPO  = os.path.expanduser("~/workspace/repos/unscan")
LOG_DEFAULT  = os.path.join(REPO, "hillclimb_log.jsonl")
LOCK_FILE    = os.path.join(REPO, "hillclimb.lock")

# ─── Parameter space ───
PARAMS = [
    ("SEAM_INK_POWER",         1.0,  [0.5, 1.5, 2.0, 3.0]),
    ("SEAM_INK_NORM",          1.0,  [0.5, 2.0, 10.0, 128.0, 255.0]),
    ("SEAM_INK_ROW_WEIGHT",    0.0,  [0.5, 1.0, 2.0, 5.0, -0.5, -1.0]),
    ("SEAM_INK_ROW_POWER",     1.0,  [0.5, 2.0]),
    ("SEAM_DELTA_WEIGHT",      4.0,  [0.0, 1.0, 2.0, 6.0, 8.0, 12.0]),
    ("SEAM_DELTA_POWER",       1.0,  [0.5, 1.5, 2.0]),
    ("SEAM_DELTA_SCALE_POWER", 1.0,  [0.0, 0.5, 2.0]),
    ("SEAM_DELTA_ROW_WEIGHT",  0.0,  [0.5, 1.0, 2.0, 5.0, -0.5, -1.0]),
    ("SEAM_DELTA_ROW_POWER",   1.0,  [0.5, 2.0]),
]

def config_key(config):
    """Deterministic hash of a config dict for dedup."""
    canon = json.dumps(config, sort_keys=True, separators=(',',':'))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]

def default_config():
    return {name: default for name, default, _ in PARAMS}

def load_history(log_file):
    """Load evaluated configs from log. Returns (history_dict, best_hits, best_config).
    
    history_dict maps config_key -> (hits, config_dict).
    best_config is the config with the highest hits (greedy-walk best, reconstructed
    by replaying the hill-climbing logic over the log so we land on the same
    current-config the live run would have reached).
    """
    history = {}   # config_key -> (hits, config_dict)
    if not os.path.exists(log_file):
        return history, -1, default_config()

    entries = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            cfg = entry.get("config", {})
            hits = entry.get("hits", -1)
            key = config_key(cfg)
            history[key] = (hits, cfg)
            entries.append(entry)

    # Replay greedy walk to reconstruct current best config
    config = default_config()
    best_hits = history.get(config_key(config), (-1, None))[0]
    if best_hits < 0:
        # Baseline wasn't the first entry; just find global best
        best_hits = max(h for h, _ in history.values())
        for key, (h, cfg) in history.items():
            if h == best_hits:
                config = dict(cfg)
                break
        return history, best_hits, config

    # Walk the same param order the hill climber uses
    changed = True
    while changed:
        changed = False
        for param_name, _default, perturbations in PARAMS:
            current_val = config[param_name]
            for new_val in perturbations:
                if abs(new_val - current_val) < 1e-9:
                    continue
                trial = dict(config)
                trial[param_name] = new_val
                trial_key = config_key(trial)
                if trial_key in history:
                    trial_hits = history[trial_key][0]
                    if trial_hits > best_hits:
                        best_hits = trial_hits
                        config[param_name] = new_val
                        changed = True
                        break  # accept first improvement, re-scan from top
            if changed:
                break

    return history, best_hits, config

def evaluate(config, eval_num, log_file):
    """Run t62 test, return (hits, compared, accuracy) or None on failure."""
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env.get("PATH", "")
    for k, v in config.items():
        env[k] = str(v)

    t0 = time.time()
    sys.stderr.write(f"  eval#{eval_num}: {json.dumps(config, separators=(',',':'))}\n")
    sys.stderr.flush()

    try:
        result = subprocess.run(
            [CARGO, "test", "--release", "--test", "t62_cross_renderer_accuracy",
             "--", "--nocapture"],
            capture_output=True, text=True, cwd=REPO, env=env, timeout=600,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"  ⚠ timeout after 600s\n")
        return None

    elapsed = time.time() - t0
    output = result.stdout + result.stderr

    m = re.search(r'Poppler AA @ 300dpi:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%', output)
    if not m:
        sys.stderr.write(f"  ⚠ parse failed (exit {result.returncode})\n")
        lines = output.strip().split('\n')
        for l in lines[-10:]:
            sys.stderr.write(f"    | {l}\n")
        return None

    hits, compared, accuracy = int(m.group(1)), int(m.group(2)), float(m.group(3))

    entry = {
        "eval": eval_num,
        "config": config,
        "hits": hits,
        "compared": compared,
        "accuracy": accuracy,
        "elapsed_s": round(elapsed, 1),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    sys.stderr.write(f"  → {hits}/{compared} = {accuracy}% ({elapsed:.0f}s)\n")
    sys.stderr.flush()
    return (hits, compared, accuracy)


def hillclimb(max_rounds, log_file):
    # ─── Resume state from log ───
    history, best_hits, config = load_history(log_file)
    eval_num = len(history)
    resumed = eval_num > 0

    print(f"=== Hill Climbing Seam Parameters ===")
    print(f"Log: {log_file}")
    print(f"Max rounds: {max_rounds}")
    if resumed:
        print(f"Resuming from eval#{eval_num}, best so far: {best_hits}/479")
        print(f"Current config: {json.dumps(config)}")
    print()

    # ─── Baseline (skip if already in log) ───
    baseline_key = config_key(default_config())
    if baseline_key not in history:
        print("─── Baseline evaluation ───")
        baseline = evaluate(default_config(), eval_num, log_file)
        eval_num += 1
        if baseline is None:
            print("FATAL: Could not get baseline score")
            return
        history[baseline_key] = (baseline[0], default_config())
        if baseline[0] > best_hits:
            best_hits = baseline[0]
            config = default_config()
        print(f"Baseline: {baseline[0]}/{baseline[1]} = {baseline[2]}%\n")
    else:
        base_hits = history[baseline_key][0]
        print(f"Baseline (from log): {base_hits}/479\n")

    for round_num in range(1, max_rounds + 1):
        print(f"═══ Round {round_num} ═══")
        improved = False

        for param_name, _default, perturbations in PARAMS:
            current_val = config[param_name]

            for new_val in perturbations:
                if abs(new_val - current_val) < 1e-9:
                    continue

                trial = dict(config)
                trial[param_name] = new_val
                trial_key = config_key(trial)

                # Skip if already evaluated
                if trial_key in history:
                    prev_hits = history[trial_key][0]
                    sys.stderr.write(f"  skip {param_name}={new_val} (cached: {prev_hits}/479)\n")
                    if prev_hits > best_hits:
                        print(f"  ★ CACHED IMPROVEMENT: {param_name}={new_val} → {prev_hits}/479 (was {best_hits})")
                        best_hits = prev_hits
                        config[param_name] = new_val
                        improved = True
                        break
                    continue

                result = evaluate(trial, eval_num, log_file)
                eval_num += 1

                if result is None:
                    continue

                hits, compared, accuracy = result
                history[trial_key] = (hits, trial)

                if hits > best_hits:
                    print(f"  ★ IMPROVEMENT: {param_name}={new_val} → {hits}/{compared} = {accuracy}% (was {best_hits})")
                    best_hits = hits
                    config[param_name] = new_val
                    improved = True
                    break

        print(f"\nEnd of round {round_num}: {best_hits}/479")
        print(f"Config: {json.dumps(config, indent=2)}")
        print()

        if not improved:
            print("No improvement found in this round. Stopping.")
            break

    print(f"\n{'='*50}")
    print(f"FINAL BEST: {best_hits}/479")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"Total evaluations: {eval_num}")
    print(f"Log: {log_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--log", default=LOG_DEFAULT)
    args = parser.parse_args()

    # ─── Lock file: prevent duplicate instances ───
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("Another hillclimb instance is already running. Exiting.")
        sys.exit(0)

    lock_fd.write(str(os.getpid()))
    lock_fd.flush()

    try:
        hillclimb(args.rounds, args.log)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        try:
            os.unlink(LOCK_FILE)
        except OSError:
            pass


if __name__ == "__main__":
    main()
