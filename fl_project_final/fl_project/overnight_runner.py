"""
overnight_runner.py - Robust overnight launcher for Experiment 3 LLM methods.

Runs: AgenticAI (Seeds 2, 3) -> SingleShotLLM (Seeds 2, 3) -> ReflectiveAgent (Seeds 2, 3)
Auto-retries on ANY crash (network errors, CUDA OOM, LLM errors, etc.)
Logs everything to overnight_runner.log
"""

import subprocess
import sys
import time
import os
import sqlite3
import logging
from datetime import datetime

# ─── Config ────────────────────────────────────────────────────────────────────
PYTHON = sys.executable
SCRIPT  = os.path.join(os.path.dirname(__file__), "run_experiment3.py")
DB_PATH = os.path.join(os.path.dirname(__file__), "fl_metrics_experiment3.db")
LOG_PATH = os.path.join(os.path.dirname(__file__), "overnight_runner.log")

# Priority order: AgenticAI first, then ablations
RUNS = [
    {"methods": ["AgenticAI"],       "seeds": [2, 3]},
    {"methods": ["SingleShotLLM"],   "seeds": [2, 3]},
    {"methods": ["ReflectiveAgent"], "seeds": [2, 3]},
]

MAX_RETRIES    = 20     # max crash-restarts per (method, seed) block
RETRY_DELAY_S  = 30     # wait 30s after a crash before restarting
NUM_ROUNDS     = 20
# ────────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def get_completed_rounds(method: str, seed: int) -> int:
    """Returns how many rounds are recorded in the DB for (method, seed)."""
    if not os.path.exists(DB_PATH):
        return 0
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            SELECT COALESCE(MAX(rd.round), 0)
            FROM experiment3_rounds rd
            JOIN experiment3_runs r ON rd.run_id = r.run_id
            WHERE r.method = ? AND r.seed = ?
        """, (method, seed))
        result = cur.fetchone()
        conn.close()
        return result[0] if result else 0
    except Exception:
        return 0


def is_complete(method: str, seed: int) -> bool:
    return get_completed_rounds(method, seed) >= NUM_ROUNDS


def run_block(methods: list, seeds: list) -> bool:
    """Run a single command block with auto-retry on crash. Returns True if all done."""
    method_str = ", ".join(methods)
    seed_str   = ", ".join(str(s) for s in seeds)

    for attempt in range(1, MAX_RETRIES + 1):
        # Check if all (method, seed) pairs are already complete — skip early
        all_done = all(is_complete(m, s) for m in methods for s in seeds)
        if all_done:
            log.info(f"  [{method_str} | Seeds {seed_str}] Already complete in DB. Skipping.")
            return True

        # Show current progress
        for m in methods:
            for s in seeds:
                done = get_completed_rounds(m, s)
                log.info(f"  Progress: {m} Seed {s} -> {done}/{NUM_ROUNDS} rounds in DB")

        cmd = [
            PYTHON, SCRIPT,
            "--seeds",   *[str(s) for s in seeds],
            "--methods", *methods,
            "--resume",
            "--no-quota-safe",
        ]
        log.info(f"\n{'='*60}")
        log.info(f"  ATTEMPT {attempt}/{MAX_RETRIES}: {method_str} | Seeds {seed_str}")
        log.info(f"  Command: {' '.join(cmd)}")
        log.info(f"{'='*60}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=os.path.dirname(__file__),
            )

            # Stream output to both log file and console
            for line in proc.stdout:
                line = line.rstrip()
                log.info(line)

            proc.wait()
            rc = proc.returncode

            if rc == 0:
                log.info(f"  SUCCESS: {method_str} | Seeds {seed_str} completed cleanly (exit code 0).")
                return True
            else:
                log.warning(f"  CRASH: Process exited with code {rc}. Retrying in {RETRY_DELAY_S}s...")

        except Exception as e:
            log.error(f"  EXCEPTION launching subprocess: {e}. Retrying in {RETRY_DELAY_S}s...")

        # After a crash, check if the DB actually has all rounds (exit code non-0 but work done)
        all_done = all(is_complete(m, s) for m in methods for s in seeds)
        if all_done:
            log.info(f"  DB shows all rounds complete despite non-zero exit. Treating as SUCCESS.")
            return True

        time.sleep(RETRY_DELAY_S)

    log.error(f"  FAILED after {MAX_RETRIES} attempts: {method_str} | Seeds {seed_str}")
    return False


def main():
    log.info("=" * 60)
    log.info(" OVERNIGHT RUNNER STARTED")
    log.info(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f" Priority order:")
    for i, run in enumerate(RUNS):
        log.info(f"   {i+1}. {run['methods']} | Seeds {run['seeds']}")
    log.info("=" * 60)

    overall_start = time.time()
    all_success = True

    for i, run in enumerate(RUNS):
        methods = run["methods"]
        seeds   = run["seeds"]
        log.info(f"\n>>> BLOCK {i+1}/{len(RUNS)}: {methods} | Seeds {seeds}")
        success = run_block(methods, seeds)
        if not success:
            log.error(f"  BLOCK FAILED: {methods}. Continuing to next block...")
            all_success = False

    elapsed = time.time() - overall_start
    log.info("\n" + "=" * 60)
    log.info(" OVERNIGHT RUNNER FINISHED")
    log.info(f" Total time: {elapsed/3600:.2f} hours")
    log.info(f" Status: {'ALL DONE' if all_success else 'SOME BLOCKS FAILED - check log'}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
