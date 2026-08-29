import os
import json
import time
import requests

from groq import Groq

from aggregation import AVAILABLE_METHODS
from database import insert_agent_log

# Safety net only: the rule-based selector and the LLM prompt never propose
# "FedProx" (it is not implemented in aggregation.py). This mapping just
# guards against the LLM naming it anyway from its own training knowledge,
# routing it to the closest implemented method instead of failing.
_METHOD_ALIASES = {
    "FedProx": "Trimmed Mean",
}


def _normalize_method(method):
    if method in AVAILABLE_METHODS:
        return method, method
    mapped = _METHOD_ALIASES.get(method)
    if mapped in AVAILABLE_METHODS:
        return mapped, method
    return AVAILABLE_METHODS[0], method


def _strip_code_fences(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    return t.strip()


def _largest_clean_subset(anomalous_ids, all_client_ids, candidate_combinations):
    anomalous_set = set(anomalous_ids)
    clean_candidates = [c for c in candidate_combinations if not (set(c) & anomalous_set)]
    if clean_candidates:
        return max(clean_candidates, key=len)
    clean_ids = [c for c in all_client_ids if c not in anomalous_set]
    return clean_ids if clean_ids else list(all_client_ids)


def _full_clean_set(anomalous_ids, all_client_ids, candidate_combinations):
    if candidate_combinations:
        return max(candidate_combinations, key=len)
    anomalous_set = set(anomalous_ids)
    clean_ids = [c for c in all_client_ids if c not in anomalous_set]
    return clean_ids if clean_ids else list(all_client_ids)


def rule_based_selector(payload):
    anomalous_ids = payload["anomalous_clients"]
    candidates = payload["candidate_combinations"]
    all_client_ids = payload.get("all_client_ids", [])
    alpha = payload.get("alpha", 1.0)

    if anomalous_ids:
        combo = _largest_clean_subset(anomalous_ids, all_client_ids, candidates)
        method = "Krum"
        reason = f"Anomalous clients {anomalous_ids} detected; excluding them and applying Byzantine-robust Krum."
    elif alpha is not None and alpha < 1.0:
        combo = _full_clean_set(anomalous_ids, all_client_ids, candidates)
        method = "Trimmed Mean"
        reason = f"High data heterogeneity (alpha={alpha}); applying robust aggregation."
    else:
        combo = _full_clean_set(anomalous_ids, all_client_ids, candidates)
        method = "FedAvg"
        reason = "No anomalies detected and low heterogeneity; FedAvg selected."

    return {"selected_combination": combo, "selected_method": method, "reason": reason, "confidence": 1.0, "signals_used": ["rule_based"]}


def _build_prompt(payload):
    return f"""You are an expert agent controlling federated learning aggregation.

Decide BOTH which client combination to aggregate and which aggregation method to use.

Input:
{json.dumps(payload, indent=2, default=str)}

Available aggregation methods: {AVAILABLE_METHODS}

Decision guidelines:
- Exclude anomalous clients whenever possible.
- Repeated anomalies -> Krum.
- Noisy clients (low cosine similarity / high weight variation) -> Median or Trimmed Mean.
- High heterogeneity (low alpha) -> Trimmed Mean (no anomalies), otherwise Krum.
- Stable clean rounds -> FedAvg.
- selected_combination MUST be exactly one of the lists in candidate_combinations.

Respond with ONLY a strict JSON object, no markdown, no preamble:
{{"selected_combination": [<client ids>], "selected_method": "<method>", "reason": "<one sentence>", "confidence": <float 0.0-1.0>, "signals_used": ["<signal1>", "<signal2>"]}}
"""


def _call_groq(prompt, model):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_ollama(prompt, model):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json().get("response", "")


def agent_select(round_num, anomalous_clients, candidate_combinations,
                  history, metrics, all_client_ids=None, alpha=None,
                  run_id=None, agent_model=None, seed=None, db_path="fl_metrics.db"):
    payload = {
        "round": round_num,
        "anomalous_clients": list(anomalous_clients),
        "candidate_combinations": [list(c) for c in candidate_combinations],
        "history": history,
        "metrics": metrics,
        "all_client_ids": list(all_client_ids) if all_client_ids is not None else [],
        "alpha": alpha,
    }

    decision, source = None, None
    raw_response = None
    prompt_version = "v2_json_with_confidence"
    latency_ms = None
    valid = False
    fallback_reason = None
    confidence = None

    if agent_model and agent_model.lower() != "rulebased":
        prompt = _build_prompt(payload)
        start_time = time.time()
        
        try:
            if agent_model.startswith("groq:"):
                model_name = agent_model.split(":", 1)[1]
                raw_response = _call_groq(prompt, model_name)
                source = agent_model
            elif agent_model.startswith("ollama:"):
                model_name = agent_model.split(":", 1)[1]
                raw_response = _call_ollama(prompt, model_name)
                source = agent_model
            else:
                raw_response = _call_ollama(prompt, agent_model)
                source = f"ollama:{agent_model}"
            
            latency_ms = (time.time() - start_time) * 1000
            
            text = _strip_code_fences(raw_response)
            parsed = json.loads(text)

            combo = parsed["selected_combination"]
            method = parsed["selected_method"]
            reason = parsed.get("reason", "")
            confidence = parsed.get("confidence")

            if not isinstance(combo, list) or not combo:
                raise ValueError(f"Invalid selected_combination: {combo}")

            valid_ids = set(payload.get("all_client_ids", []))
            if valid_ids and not set(combo).issubset(valid_ids):
                raise ValueError(f"Unknown client ids in combination: {combo}")

            decision = {
                "selected_combination": combo, 
                "selected_method": method, 
                "reason": reason,
                "confidence": confidence,
            }
            valid = True

        except Exception as e:
            if latency_ms is None:
                latency_ms = (time.time() - start_time) * 1000
            fallback_reason = str(e)
            print(f"  [Agent] Backend {agent_model} call failed ({e}); using rule-based fallback.")

    if decision is None:
        decision = rule_based_selector(payload)
        source = "rule_based"
        confidence = decision.get("confidence", 1.0)
        if agent_model and agent_model.lower() != "rulebased":
            valid = False
        else:
            valid = True

    final_method, proposed_method = _normalize_method(decision["selected_method"])
    reason = decision["reason"]
    if final_method != proposed_method:
        reason = f"{reason} ('{proposed_method}' mapped to '{final_method}')"

    combo = decision["selected_combination"]
    combo_str = "-".join(f"C{cid}" for cid in combo)
    print(f"  [Agent:{source}] Method={final_method} | Combination ({combo_str}) | {reason}")

    insert_agent_log(run_id, round_num, alpha, final_method, reason, source,
                      selected_combination=combo, 
                      raw_response=raw_response, 
                      prompt_version=prompt_version, 
                      latency_ms=latency_ms, 
                      valid=int(valid) if type(valid) == bool else valid, 
                      fallback_reason=fallback_reason, 
                      confidence=confidence,
                      seed=seed, 
                      db_path=db_path)

    return {
        "selected_combination": combo,
        "selected_method": final_method,
        "reason": reason,
        "source": source,
        "confidence": confidence
    }