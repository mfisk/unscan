#!/usr/bin/env python3
"""
CMA-ES multidimensional parameter search for seam carving scoring.
Resumable: reads hillclimb_log.jsonl to skip already-evaluated configs.
Uses a lock file to prevent duplicate instances.

Searches all 9 parameters simultaneously using covariance matrix adaptation,
which learns parameter correlations and explores diagonal ridges that
coordinate-wise hill climbing misses.

Parameters (env vars fed to the binary):
  SEAM_INK_POWER, SEAM_INK_NORM, SEAM_INK_ROW_WEIGHT, SEAM_INK_ROW_POWER,
  SEAM_DELTA_WEIGHT, SEAM_DELTA_POWER, SEAM_DELTA_SCALE_POWER,
  SEAM_DELTA_ROW_WEIGHT, SEAM_DELTA_ROW_POWER
"""

import subprocess, re, os, sys, json, time, fcntl, hashlib, argparse
import numpy as np
from cmaes import CMA, get_warm_start_mgd

CARGO = os.path.expanduser("~/.cargo/bin/cargo")
REPO  = os.path.expanduser("~/workspace/repos/unscan")
LOG_DEFAULT  = os.path.join(REPO, "hillclimb_log.jsonl")
LOCK_FILE    = os.path.join(REPO, "hillclimb.lock")

# ─── Parameter definitions ───
# (name, default, lower_bound, upper_bound)
PARAMS = [
    ("SEAM_INK_POWER",         1.0,   0.1,  4.0),
    ("SEAM_INK_NORM",          1.0,   0.1, 255.0),
    ("SEAM_INK_ROW_WEIGHT",    0.0,  -2.0,  10.0),
    ("SEAM_INK_ROW_POWER",     1.0,   0.1,  4.0),
    ("SEAM_DELTA_WEIGHT",      4.0,   0.0,  20.0),
    ("SEAM_DELTA_POWER",       1.0,   0.1,  4.0),
    ("SEAM_DELTA_SCALE_POWER", 1.0,   0.0,  4.0),
    ("SEAM_DELTA_ROW_WEIGHT",  0.0,  -2.0,  10.0),
    ("SEAM_DELTA_ROW_POWER",   1.0,   0.1,  4.0),
]

PARAM_NAMES  = [p[0] for p in PARAMS]
PARAM_DEFAULTS = np.array([p[1] for p in PARAMS])
PARAM_BOUNDS = np.array([[p[2], p[3]] for p in PARAMS])
N_PARAMS = len(PARAMS)


def config_key(config):
    """Deterministic hash of a config dict for dedup."""
    # Round to 4 decimal places for dedup (CMA samples are continuous)
    rounded = {k: round(v, 4) for k, v in config.items()}
    canon = json.dumps(rounded, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def vec_to_config(x):
    """Convert numpy vector to config dict."""
    return {name: float(x[i]) for i, name in enumerate(PARAM_NAMES)}


def config_to_vec(config):
    """Convert config dict to numpy vector."""
    return np.array([config[name] for name in PARAM_NAMES])


def load_history(log_file):
    """Load all evaluated configs from the JSONL log.
    Returns: (list of (vec, score) tuples, dict[config_key -> hits])"""
    history = {}
    solutions = []
    if not os.path.exists(log_file):
        return solutions, history

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
            history[key] = hits
            vec = config_to_vec(cfg)
            solutions.append((vec, hits))
    return solutions, history


def evaluate(config, eval_num, log_file):
    """Run t62 test, return (hits, compared, accuracy) or None on failure."""
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env.get("PATH", "")
    for k, v in config.items():
        env[k] = str(v)

    t0 = time.time()
    sys.stderr.write(f"  eval#{eval_num}: {json.dumps({k: round(v,3) for k,v in config.items()}, separators=(',',':'))}\n")
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
        "method": "cma-es",
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    sys.stderr.write(f"  → {hits}/{compared} = {accuracy}% ({elapsed:.0f}s)\n")
    sys.stderr.flush()
    return (hits, compared, accuracy)


def cmaes_search(max_evals, log_file, seed=42):
    """Run CMA-ES search over the 9 seam parameters."""

    # ─── Load history for warm start and dedup ───
    past_solutions, history = load_history(log_file)
    eval_num = len(history)
    best_hits = max((h for h in history.values()), default=-1)

    print(f"=== CMA-ES Seam Parameter Search ===")
    print(f"Log: {log_file}")
    print(f"Max evaluations: {max_evals}")
    print(f"Parameters: {N_PARAMS}")
    print(f"Prior evaluations: {eval_num}")
    if best_hits >= 0:
        print(f"Prior best: {best_hits}/479")
    print()

    # ─── Initialize CMA-ES ───
    if past_solutions and len(past_solutions) >= 5:
        # Warm start from prior evaluations
        # CMA-ES minimizes, so negate hits (we want to maximize)
        source = [(vec, -score) for vec, score in past_solutions]
        mean, sigma, cov = get_warm_start_mgd(source, gamma=0.2, alpha=0.1)
        # Clip mean to bounds
        mean = np.clip(mean, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
        print(f"Warm-started from {len(past_solutions)} prior evals")
        print(f"  Initial mean: {dict(zip(PARAM_NAMES, [round(v,3) for v in mean]))}")
        print(f"  Initial sigma: {sigma:.4f}")
        optimizer = CMA(
            mean=mean,
            sigma=sigma,
            cov=cov,
            bounds=PARAM_BOUNDS,
            seed=seed,
            lr_adapt=True,  # adaptive learning rate for noisy/challenging landscapes
        )
    else:
        # Start from defaults
        optimizer = CMA(
            mean=PARAM_DEFAULTS.copy(),
            sigma=2.0,
            bounds=PARAM_BOUNDS,
            seed=seed,
            lr_adapt=True,
        )
        print("Starting from default parameters")

    pop_size = optimizer.population_size
    print(f"Population size: {pop_size}")
    print()

    evals_done = 0
    generation = 0
    stagnation_count = 0
    last_best = best_hits

    while evals_done < max_evals:
        generation += 1
        print(f"═══ Generation {generation} (evals {eval_num}–{eval_num + pop_size - 1}) ═══")

        solutions = []  # list of (x, value) for CMA tell()

        for _ in range(pop_size):
            if evals_done >= max_evals:
                break

            x = optimizer.ask()
            config = vec_to_config(x)
            key = config_key(config)

            # Check cache
            if key in history:
                cached_hits = history[key]
                sys.stderr.write(f"  cache hit: {cached_hits}/479\n")
                solutions.append((x, -cached_hits))  # negate for minimization
                continue

            result = evaluate(config, eval_num, log_file)
            eval_num += 1
            evals_done += 1

            if result is None:
                # Treat failures as worst possible score
                solutions.append((x, 0.0))
                history[key] = 0
                continue

            hits, compared, accuracy = result
            solutions.append((x, -hits))  # negate for minimization
            history[key] = hits

            if hits > best_hits:
                best_hits = hits
                print(f"  ★ NEW BEST: {hits}/{compared} = {accuracy}%")
                print(f"    Config: {json.dumps({k: round(v,3) for k,v in config.items()}, indent=2)}")

        if not solutions:
            break

        optimizer.tell(solutions)

        # Generation summary
        gen_scores = [-v for _, v in solutions]
        print(f"  Generation {generation}: min={min(gen_scores):.0f} mean={np.mean(gen_scores):.1f} max={max(gen_scores):.0f} best_ever={best_hits}")

        # Check for convergence/stagnation
        if best_hits == last_best:
            stagnation_count += 1
        else:
            stagnation_count = 0
            last_best = best_hits

        if stagnation_count >= 10:
            print(f"\nStagnation detected ({stagnation_count} generations without improvement)")
            print("Consider restarting with BIPOP or different sigma")
            break

        if optimizer.should_stop():
            print(f"\nCMA-ES internal stopping criterion met")
            break

        print()

    # ─── Final report ───
    # Find the actual best config
    best_vec = None
    for vec, score in past_solutions:
        if history.get(config_key(vec_to_config(vec)), -1) == best_hits:
            best_vec = vec
    # Also check new evals
    with open(log_file) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("hits", -1) == best_hits:
                best_config = entry["config"]
                break

    print(f"\n{'='*50}")
    print(f"FINAL BEST: {best_hits}/479")
    print(f"Config: {json.dumps(best_config, indent=2)}")
    print(f"Total evaluations: {eval_num} ({evals_done} new)")
    print(f"Generations: {generation}")
    print(f"Log: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="CMA-ES seam parameter search")
    parser.add_argument("--max-evals", type=int, default=200,
                        help="Maximum new evaluations (default: 200)")
    parser.add_argument("--log", default=LOG_DEFAULT,
                        help="JSONL log file (shared with hillclimb.py)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ─── Lock file: prevent duplicate instances ───
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("Another search instance is already running. Exiting.")
        sys.exit(0)

    lock_fd.write(str(os.getpid()))
    lock_fd.flush()

    try:
        cmaes_search(args.max_evals, args.log, args.seed)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        try:
            os.unlink(LOCK_FILE)
        except OSError:
            pass


if __name__ == "__main__":
    main()
