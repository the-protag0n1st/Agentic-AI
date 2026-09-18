import argparse
import os
import sqlite3
import sys
import time

DB_NAME = "fl_metrics_experiment3.db"

def get_db_path():
    candidates = [
        os.path.join(os.getcwd(), DB_NAME),
        os.path.join(os.path.dirname(__file__), DB_NAME),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), DB_NAME),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

def print_progress():
    db_path = get_db_path()
    if not os.path.exists(db_path):
        print(f"[-] Database not found at: {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiment3_runs'")
        if not cur.fetchone():
            print(f"[-] Table 'experiment3_runs' not found in {db_path}")
            conn.close()
            return

        print("\n" + "=" * 70)
        print("  FEDERATED LEARNING - EXPERIMENT 3 STATUS")
        print("=" * 70)

        cur.execute('''
            SELECT r.method, r.seed, COALESCE(MAX(rd.round), 0) as max_round, COUNT(rd.round) as cnt
            FROM experiment3_runs r
            LEFT JOIN experiment3_rounds rd ON r.run_id = rd.run_id
            GROUP BY r.method, r.seed
            ORDER BY r.method, r.seed
        ''')
        rows = cur.fetchall()

        print(f" {'Method':<20} | {'Seed':<6} | {'Completed Rounds':<18} | {'Status'}")
        print(" " + "-" * 66)
        for method, seed, max_rnd, cnt in rows:
            status = "[DONE]" if max_rnd == 20 else f"[RUNNING {max_rnd}/20]"
            print(f" {method:<20} | {seed:<6} | {max_rnd:>2}/20 rounds      | {status}")

        cur.execute('''
            SELECT r.method, r.seed, rd.round, rd.alpha, rd.val_accuracy, rd.test_accuracy, rd.round_time_s
            FROM experiment3_rounds rd
            JOIN experiment3_runs r ON rd.run_id = r.run_id
            ORDER BY rd.id DESC
            LIMIT 5
        ''')
        recent = cur.fetchall()
        if recent:
            print("\n" + "=" * 70)
            print("  RECENT ROUNDS COMPLETED")
            print("=" * 70)
            print(f" {'Method':<18} | {'Seed':<5} | {'Round':<6} | {'Alpha':<6} | {'Val Acc':<8} | {'Test Acc':<8} | {'Time (s)'}")
            print(" " + "-" * 66)
            for r in recent:
                val_acc = f"{r[4]:.2f}%" if r[4] is not None and r[4] > 1.0 else (f"{r[4]*100:.2f}%" if r[4] else "N/A")
                test_acc = f"{r[5]:.2f}%" if r[5] is not None and r[5] > 1.0 else (f"{r[5]*100:.2f}%" if r[5] else "N/A")
                time_s = f"{r[6]:.1f}s" if r[6] else "N/A"
                print(f" {r[0]:<18} | {r[1]:<5} | {r[2]:<6} | {r[3]:<6.1f} | {val_acc:<8} | {test_acc:<8} | {time_s}")

        conn.close()
    except Exception as e:
        print(f"[-] Error querying database: {e}")

def main():
    parser = argparse.ArgumentParser(description="Monitor Experiment 3 progress")
    parser.add_argument("--watch", action="store_true", help="Continuously refresh progress")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval in seconds (default: 10)")
    args = parser.parse_args()

    if args.watch:
        print(f"Watching progress every {args.interval}s (Press Ctrl+C to stop)...")
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                print_progress()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped monitor.")
    else:
        print_progress()

if __name__ == "__main__":
    main()
