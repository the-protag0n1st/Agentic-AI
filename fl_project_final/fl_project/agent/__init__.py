"""Reflective Agentic AI Package for Adaptive Federated Learning.

Exports the complete reflective multi-stage agentic decision architecture
while maintaining 100% backward compatibility with legacy single-shot callers.
"""

import importlib.util
import os
from pathlib import Path

# --- New Reflective Agentic API ---
from .agentic_ai import ReflectiveAgent
from .agentic_controller import AgenticController
from .analyst import Analyst
from .critic import Critic
from .history import AgentHistory
from .llm_client import GroqLLM, LLMClient, MockLLM, OllamaLLM
from .outcome_memory import ObservableOutcome, OutcomeMemory
from .proposer import Proposer
from .single_shot import SingleShotAgent
from .tools import ToolCall, ToolRegistry, ToolResult
from .schemas import (
    AgentResult,
    AgentTiming,
    AnalystOutput,
    ClientTelemetry,
    CriticOutput,
    Decision,
    Proposal,
    ProposerOutput,
    RoundTelemetry,
)
from .validation import (
    check_mathematical_feasibility,
    repair_decision,
    validate_decision,
)

# --- Backward Compatibility Layer for Legacy agent.py callers ---
# Ensures existing scripts doing `from agent import agent_select` continue to work seamlessly.
_legacy_agent_path = Path(__file__).resolve().parent.parent / "agent.py"
if _legacy_agent_path.exists():
    try:
        _spec = importlib.util.spec_from_file_location("legacy_agent_module", _legacy_agent_path)
        if _spec and _spec.loader:
            _legacy_mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_legacy_mod)
            agent_select = getattr(_legacy_mod, "agent_select", None)
            rule_based_selector = getattr(_legacy_mod, "rule_based_selector", None)
    except Exception:
        pass

__all__ = [
    "AgenticController",
    "ReflectiveAgent",
    "SingleShotAgent",
    "Analyst",
    "Proposer",
    "Critic",
    "Decision",
    "ClientTelemetry",
    "RoundTelemetry",
    "AnalystOutput",
    "Proposal",
    "ProposerOutput",
    "CriticOutput",
    "AgentResult",
    "AgentTiming",
    "AgentHistory",
    "ObservableOutcome",
    "OutcomeMemory",
    "ToolRegistry",
    "ToolCall",
    "ToolResult",
    "check_mathematical_feasibility",
    "validate_decision",
    "repair_decision",
    "LLMClient",
    "MockLLM",
    "GroqLLM",
    "OllamaLLM",
    # Legacy exports
    "agent_select",
    "rule_based_selector",
]
