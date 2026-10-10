import csv
from collections import defaultdict

# Read runs
with open(r'results\experiment3_runs.csv', 'r') as f:
    reader = csv.DictReader(f)
    runs = list(reader)

print("=== RUNS ===")
for r in runs:
    print(f"  run_id={r['run_id']:40s} method={r['method']:20s} seed={r['seed']}")

print(f"\nTotal run records: {len(runs)}")

# Read rounds
with open(r'results\experiment3_rounds.csv', 'r') as f:
    reader = csv.DictReader(f)
    rounds_data = list(reader)

print(f"Total round records: {len(rounds_data)}")

# Count per run_id
run_id_counts = defaultdict(list)
for rd in rounds_data:
    run_id_counts[rd['run_id']].append(int(rd['round']))

print("\n=== ROUNDS PER RUN_ID ===")
for rid, rnds in sorted(run_id_counts.items()):
    print(f"  {rid:50s} rounds={len(rnds)}/20  range=[{min(rnds)}-{max(rnds)}]")

# Now count by method across all seeds
method_rounds = defaultdict(lambda: defaultdict(list))
for rd in rounds_data:
    rid = rd['run_id']
    # Parse method and seed from run_id
    # Format: exp3_{method}_s{seed}_{timestamp}
    parts = rid.split('_s')
    method = parts[0].replace('exp3_', '')
    seed = parts[1].split('_')[0] if len(parts) > 1 else '?'
    method_rounds[method][seed].append(int(rd['round']))

print("\n=== SUMMARY BY METHOD x SEED ===")
expected_methods = ['FixedFedAvg', 'FixedMedian', 'FixedTrimmedMean', 'FixedKrum',
                    'RuleBased', 'SingleShotLLM', 'ReflectiveAgent', 'AgenticAI']
expected_seeds = ['1', '2', '3', '4', '5', '6', '7', '8']

all_complete = True
for m in expected_methods:
    for s in expected_seeds:
        rnds = method_rounds.get(m, {}).get(s, [])
        status = "COMPLETE" if len(rnds) >= 20 else f"INCOMPLETE ({len(rnds)}/20)"
        if len(rnds) < 20:
            all_complete = False
        print(f"  {m:20s} seed={s}  {status}")

print(f"\n{'ALL ROUNDS COMPLETE!' if all_complete else 'SOME ROUNDS MISSING'}")

# Print sample data
print("\n=== SAMPLE ROUND DATA (last 3 rows) ===")
for rd in rounds_data[-3:]:
    print(f"  run_id={rd['run_id']} round={rd['round']} test_acc={rd['test_accuracy']} test_f1={rd['test_f1']} oracle_acc={rd['oracle_accuracy']} regret={rd['regret']} cumul_regret={rd['cumulative_regret']}")
