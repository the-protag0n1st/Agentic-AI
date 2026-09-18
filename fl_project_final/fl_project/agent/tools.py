"""Agentic Tool System with Real Execution Loop.

Defines callable tools that the LLM can invoke during reasoning.
Each tool is a deterministic Python function — the LLM cannot execute arbitrary code.
Tool calls trigger actual Python execution, return structured results,
and allow the LLM to use the result in subsequent reasoning steps.
"""

import json
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from .schemas import ClientTelemetry, Decision
from .history import AgentHistory


class ToolSpec(BaseModel):
    """Schema describing a tool's interface for the LLM."""
    name: str
    description: str
    parameters: Dict[str, str] = Field(default_factory=dict)
    returns: str = ""


class ToolCall(BaseModel):
    """A single tool invocation requested by the LLM."""
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of executing a tool."""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None


class ToolRegistry:
    """Registry of callable tools with execution and budget enforcement.
    
    Each tool is bound to the current round's context (history, candidates,
    telemetry, outcome memory) at construction time.
    """
    
    def __init__(
        self,
        history: AgentHistory,
        candidate_map: Dict[str, Decision],
        outcome_memory: Any,  # OutcomeMemory, avoid circular import
        telemetry: List[ClientTelemetry],
        previous_telemetry: Optional[List[ClientTelemetry]] = None,
        max_tool_calls: int = 5,
    ):
        self._tools: Dict[str, Callable] = {}
        self._specs: Dict[str, ToolSpec] = {}
        self._call_log: List[str] = []
        self._max_tool_calls = max_tool_calls
        
        # Bind context
        self._history = history
        self._candidate_map = candidate_map
        self._outcome_memory = outcome_memory
        self._telemetry = telemetry
        self._previous_telemetry = previous_telemetry
        
        # Register all tools
        self._register_builtin_tools()
    
    def _register_builtin_tools(self) -> None:
        """Register the 5 built-in tools."""
        self._register(
            name="get_client_trajectory",
            func=self._tool_get_client_trajectory,
            spec=ToolSpec(
                name="get_client_trajectory",
                description="Get the multi-round telemetry trajectory for a specific client. Use this to investigate suspicious clients.",
                parameters={"client_id": "int: The client ID to investigate", "num_rounds": "int: Number of recent rounds (default 3)"},
                returns="Dict with rounds of norm, cosine, flagged status, weight variance for the client.",
            ),
        )
        self._register(
            name="check_candidate_feasibility",
            func=self._tool_check_feasibility,
            spec=ToolSpec(
                name="check_candidate_feasibility",
                description="Check if a specific candidate decision is mathematically feasible for its aggregation method.",
                parameters={"candidate_id": "str: The candidate ID to check (e.g. C0, C1)"},
                returns="Dict with feasible (bool), reason, method, num_clients, client_ids.",
            ),
        )
        self._register(
            name="compare_candidates",
            func=self._tool_compare_candidates,
            spec=ToolSpec(
                name="compare_candidates",
                description="Compare two candidate decisions side-by-side, showing shared and different clients and methods.",
                parameters={"candidate_id_a": "str: First candidate ID", "candidate_id_b": "str: Second candidate ID"},
                returns="Dict with shared_clients, only_in_a, only_in_b, method_a, method_b, set_difference_size.",
            ),
        )
        self._register(
            name="get_outcome_history",
            func=self._tool_get_outcome_history,
            spec=ToolSpec(
                name="get_outcome_history",
                description="Get the observable outcomes from recent rounds (accuracy, loss, method used, anomaly stats). Does NOT include Oracle data.",
                parameters={"num_rounds": "int: Number of recent rounds (default 3)"},
                returns="List of dicts with round, selected_method, observed_accuracy, accuracy_delta, flagged_client_count.",
            ),
        )
        self._register(
            name="get_regime_indicators",
            func=self._tool_get_regime_indicators,
            spec=ToolSpec(
                name="get_regime_indicators",
                description="Get computed statistical indicators from current telemetry for operational regime assessment. Uses only observable statistics.",
                parameters={},
                returns="Dict with mean_cosine, cosine_variance, mean_norm, norm_variance, flagged_ratio, client_disagreement, update_drift, accuracy_trend.",
            ),
        )
    
    def _register(self, name: str, func: Callable, spec: ToolSpec) -> None:
        self._tools[name] = func
        self._specs[name] = spec
    
    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call. Returns error if budget exhausted or tool unknown."""
        if len(self._call_log) >= self._max_tool_calls:
            return ToolResult(
                tool_name=tool_call.name,
                success=False,
                error=f"Tool budget exhausted ({self._max_tool_calls}/{self._max_tool_calls} used). Produce your final answer now.",
            )
        
        if tool_call.name not in self._tools:
            return ToolResult(
                tool_name=tool_call.name,
                success=False,
                error=f"Unknown tool '{tool_call.name}'. Available: {list(self._tools.keys())}",
            )
        
        try:
            result = self._tools[tool_call.name](**tool_call.args)
            self._call_log.append(tool_call.name)
            return ToolResult(tool_name=tool_call.name, success=True, result=result)
        except Exception as e:
            self._call_log.append(f"{tool_call.name}(error)")
            return ToolResult(tool_name=tool_call.name, success=False, error=str(e))
    
    @property
    def budget_remaining(self) -> int:
        return max(0, self._max_tool_calls - len(self._call_log))
    
    @property
    def call_log(self) -> List[str]:
        return list(self._call_log)
    
    def get_tool_descriptions(self) -> str:
        """Format all tool specs as text for LLM prompt injection."""
        lines = []
        for name, spec in self._specs.items():
            params_str = ", ".join(f"{k}: {v}" for k, v in spec.parameters.items())
            lines.append(f"- {spec.name}({params_str})")
            lines.append(f"  {spec.description}")
            lines.append(f"  Returns: {spec.returns}")
            lines.append("")
        return "\n".join(lines)
    
    # ========== Tool Implementations ==========
    
    def _tool_get_client_trajectory(self, client_id: int, num_rounds: int = 3) -> Dict:
        """Tool 1: Get client trajectory from AgentHistory."""
        trajectory = self._history.get_client_trajectory(int(client_id), n=int(num_rounds))
        persistent = self._history.has_persistent_anomaly(int(client_id), min_rounds=2)
        return {
            "client_id": int(client_id),
            "trajectory": trajectory,
            "persistent_anomaly": persistent,
            "rounds_available": len(trajectory),
        }
    
    def _tool_check_feasibility(self, candidate_id: str) -> Dict:
        """Tool 2: Check candidate feasibility."""
        candidate = self._candidate_map.get(str(candidate_id))
        if candidate is None:
            return {
                "candidate_id": str(candidate_id),
                "feasible": False,
                "reason": f"Candidate '{candidate_id}' not found in candidate pool.",
            }
        
        from .validation import check_mathematical_feasibility
        is_feasible, reason = check_mathematical_feasibility(
            candidate.method, candidate.client_ids
        )
        return {
            "candidate_id": str(candidate_id),
            "feasible": is_feasible,
            "reason": reason,
            "method": candidate.method,
            "num_clients": len(candidate.client_ids),
            "client_ids": sorted(candidate.client_ids),
        }
    
    def _tool_compare_candidates(self, candidate_id_a: str, candidate_id_b: str) -> Dict:
        """Tool 3: Compare two candidates."""
        cand_a = self._candidate_map.get(str(candidate_id_a))
        cand_b = self._candidate_map.get(str(candidate_id_b))
        
        if cand_a is None or cand_b is None:
            missing = []
            if cand_a is None:
                missing.append(str(candidate_id_a))
            if cand_b is None:
                missing.append(str(candidate_id_b))
            return {"error": f"Candidate(s) not found: {missing}"}
        
        set_a = set(cand_a.client_ids)
        set_b = set(cand_b.client_ids)
        
        return {
            "candidate_a": str(candidate_id_a),
            "candidate_b": str(candidate_id_b),
            "method_a": cand_a.method,
            "method_b": cand_b.method,
            "clients_a": sorted(cand_a.client_ids),
            "clients_b": sorted(cand_b.client_ids),
            "shared_clients": sorted(set_a & set_b),
            "only_in_a": sorted(set_a - set_b),
            "only_in_b": sorted(set_b - set_a),
            "set_difference_size": len(set_a.symmetric_difference(set_b)),
        }
    
    def _tool_get_outcome_history(self, num_rounds: int = 3) -> List[Dict]:
        """Tool 4: Get observable outcome history. NO Oracle data."""
        return self._outcome_memory.get_history_dicts(n=int(num_rounds))
    
    def _tool_get_regime_indicators(self) -> Dict:
        """Tool 5: Compute regime indicators from observable telemetry.
        
        Does NOT expose Dirichlet alpha, attacker IDs, or any experimental config.
        Uses only statistics computable from client telemetry.
        """
        import numpy as np
        
        # Current round statistics
        cosines = [c.cosine_similarity for c in self._telemetry if c.cosine_similarity is not None]
        norms = [c.update_norm for c in self._telemetry if c.update_norm is not None]
        flagged = [c for c in self._telemetry if c.is_flagged_by_detector is True]
        
        mean_cosine = float(np.mean(cosines)) if cosines else 0.0
        cosine_var = float(np.var(cosines)) if len(cosines) > 1 else 0.0
        mean_norm = float(np.mean(norms)) if norms else 0.0
        norm_var = float(np.var(norms)) if len(norms) > 1 else 0.0
        flagged_ratio = len(flagged) / max(len(self._telemetry), 1)
        
        # Client disagreement: std of pairwise cosine similarities
        client_disagreement = float(np.std(cosines)) if len(cosines) > 1 else 0.0
        
        # Update drift: change in mean cosine vs previous round
        update_drift = 0.0
        if self._previous_telemetry:
            prev_cosines = [c.cosine_similarity for c in self._previous_telemetry if c.cosine_similarity is not None]
            if prev_cosines:
                prev_mean = float(np.mean(prev_cosines))
                update_drift = mean_cosine - prev_mean
        
        # Accuracy trend from outcome memory
        accuracy_trend = 0.0
        recent = self._outcome_memory.get_recent(2)
        if len(recent) >= 2:
            accuracy_trend = recent[-1].observed_accuracy - recent[-2].observed_accuracy
        
        # Autonomous regime classification from observables
        if flagged_ratio >= 0.2 and mean_cosine < 0.3:
            assessed_regime = "possible_attack"
        elif cosine_var > 0.1 or client_disagreement > 0.4:
            assessed_regime = "highly_heterogeneous"
        elif cosine_var > 0.03 or client_disagreement > 0.2:
            assessed_regime = "heterogeneous"
        else:
            assessed_regime = "near_iid"
        
        return {
            "mean_cosine_similarity": round(mean_cosine, 4),
            "cosine_variance": round(cosine_var, 4),
            "mean_update_norm": round(mean_norm, 4),
            "norm_variance": round(norm_var, 4),
            "flagged_client_ratio": round(flagged_ratio, 4),
            "client_disagreement": round(client_disagreement, 4),
            "update_drift": round(update_drift, 4),
            "accuracy_trend": round(accuracy_trend, 4),
            "assessed_regime": assessed_regime,
        }


def parse_tool_calls(response_text: str) -> List[ToolCall]:
    """Parse tool_calls from an LLM JSON response.
    
    Expects the LLM response to contain a JSON object with an optional
    "tool_calls" field: [{"name": "...", "args": {...}}, ...]
    
    Returns empty list if no tool_calls found or parsing fails.
    """
    try:
        # Clean markdown fences if present
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        
        data = json.loads(text)
        raw_calls = data.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            return []
        
        calls = []
        for tc in raw_calls:
            if isinstance(tc, dict) and "name" in tc:
                calls.append(ToolCall(
                    name=tc["name"],
                    args=tc.get("args", {}),
                ))
        return calls
    except (json.JSONDecodeError, Exception):
        return []
