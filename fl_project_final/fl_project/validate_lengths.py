import sqlite3
import json

db_path = "fl_metrics_experiment1_90runs_baseline.db"

con = sqlite3.connect(db_path)
cur = con.cursor()
rows = cur.execute("SELECT raw_response FROM agent_log WHERE source LIKE 'ollama%' AND raw_response IS NOT NULL").fetchall()

lengths_chars = []
lengths_tokens = []
truncation_flags = []

for r in rows:
    resp = r[0]
    lengths_chars.append(len(resp))
    lengths_tokens.append(len(resp) / 4.0) # Approx token count
    
    # Check if JSON parsing would succeed (just in case)
    try:
        j = json.loads(resp)
        # Check if 'reason' is present and unusually short which might imply truncation if there's no closing bracket
    except json.JSONDecodeError:
        truncation_flags.append(True)

print(f"Total LLM responses analyzed: {len(rows)}")
print(f"Average length (chars): {sum(lengths_chars)/len(lengths_chars):.1f}")
print(f"Max length (chars): {max(lengths_chars)}")
print(f"Average approx tokens: {sum(lengths_tokens)/len(lengths_tokens):.1f}")
print(f"Max approx tokens: {max(lengths_tokens):.1f}")
print(f"Responses failing JSON decode (likely truncated): {len(truncation_flags)}")

con.close()
