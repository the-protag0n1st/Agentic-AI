"""Prompt Templates for the Agentic Decision Controller.

Defines prompts for outcome-aware reflection, regime assessment,
tool-augmented reasoning, structured critic validation, and
Proposer revision on Critic rejection.

These prompts supplement (not replace) the existing stage prompts in prompts.py.
"""

# --- Observable Outcome Reflection (NO Oracle data) ---
OUTCOME_REFLECTION_TEMPLATE = """=== OUTCOME HISTORY (Observable Post-Decision Information) ===
{reflection_text}"""


# --- Regime Assessment (Observable Statistics Only) ---
REGIME_ASSESSMENT_TEMPLATE = """=== OPERATIONAL REGIME ASSESSMENT ===
Observable Indicators:
- Mean Cosine Similarity: {mean_cosine:.4f} (variance: {cosine_var:.4f})
- Mean Update Norm: {mean_norm:.4f} (variance: {norm_var:.4f})
- Flagged Client Ratio: {flagged_ratio:.1%}
- Client Disagreement: {client_disagreement:.4f}
- Update Drift (vs. previous round): {update_drift:+.4f}
- Accuracy Trend: {accuracy_trend:+.4f}
Assessed Regime: {assessed_regime}"""


# --- Tool Use Instructions ---
TOOL_USE_ADDENDUM = """You have access to investigative tools. To use a tool, include "tool_calls" in your JSON response.

Available Tools:
{tool_descriptions}

To invoke tools, add to your JSON response:
  "tool_calls": [{{"name": "<tool_name>", "args": {{...}}}}]

After tool execution, you will receive the results and may continue reasoning.
You have {tool_budget_remaining} remaining tool call(s).
When you have gathered sufficient evidence, omit "tool_calls" and provide your final structured response."""


# --- Tool Results Injection ---
TOOL_RESULTS_TEMPLATE = """=== TOOL RESULTS ===
{tool_results_text}

You have {tool_budget_remaining} remaining tool call(s).
If you need more information, include additional "tool_calls".
Otherwise, provide your final structured response now."""


# --- Critic Structured Validation Prompt ---
CRITIC_VALIDATION_SYSTEM = """You are an adversarial Federated Learning Critic & Validator.

Your role is to rigorously evaluate the Proposer's recommendation against
observable evidence and mathematical constraints.

You MUST evaluate each of these criteria EXPLICITLY:
1. feasible: Is the candidate mathematically feasible for its aggregation method?
2. uses_valid_clients: Are all proposed client IDs legitimate participants?
3. respects_byzantine_constraints: Does the method satisfy its theoretical requirements?
   (e.g., Krum needs n >= 2m + 3; TrimmedMean needs n >= 2k + 1)
4. addresses_anomaly_evidence: Does the decision account for detected anomalies?
5. evidence_consistent: Is the justification supported by observable telemetry?

List ALL unresolved concerns. An empty list signals acceptance.
Confidence is a supplementary annotation, NOT the acceptance criterion.

You MUST respond with ONLY a valid, parseable JSON object. No markdown fences, no preamble."""

CRITIC_VALIDATION_USER = """Communication Round: {round_num}

{regime_assessment}

Analyst Summary:
- Risk Level: {risk_level}
- Suspected Byzantine: {suspected_byzantine}
- Heterogeneity Clients: {heterogeneity_clients}
- Synthesis: {signal_interpretation}

Proposal Under Review:
{proposal_text}

Full Feasible Candidate Set:
{candidate_list}

Client Telemetry & History:
{telemetry_summary}

{outcome_reflection}

Evaluate the proposal against ALL criteria and respond with:
{{
  "recommended_candidate_id": "<exact candidate ID from the list>",
  "feasible": true,
  "uses_valid_clients": true,
  "respects_byzantine_constraints": true,
  "addresses_anomaly_evidence": true,
  "evidence_consistent": true,
  "unresolved_concerns": [],
  "critique": "<rigorous evaluation>",
  "confidence": 0.85
}}"""


# --- Proposer Revision Prompt (on Critic Rejection) ---
PROPOSER_REVISION_ADDENDUM = """=== REVISION REQUIRED ===
Your previous proposal was REJECTED by the Critic.

Failing Checks:
{failing_checks}

Critique:
{previous_critique}

Unresolved Concerns:
{unresolved_concerns}

Previously Proposed (DO NOT repeat): {rejected_ids}

You MUST propose DIFFERENT candidates that address the above concerns.
Select ONLY from the feasible candidate set provided."""


# --- Analyst Enhanced System Prompt (adds to existing ANALYST_SYSTEM_PROMPT) ---
ANALYST_AGENTIC_ADDENDUM = """Additionally, you have access to the following context:

{outcome_reflection}

{regime_assessment}

{tool_instructions}"""
