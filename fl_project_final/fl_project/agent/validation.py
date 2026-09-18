"""Deterministic Validation and Feasibility Filter.

Ensures the LLM NEVER executes an arbitrary, unfeasible, or invented aggregation decision.
All decisions must satisfy mathematical constraints and match pre-approved candidate sets.
Directly aligns with feasibility.py.
"""

from feasibility import (
    SUPPORTED_METHODS,
    check_mathematical_feasibility,
    validate_decision,
    repair_decision,
)
from .schemas import Decision

__all__ = [
    "SUPPORTED_METHODS",
    "check_mathematical_feasibility",
    "validate_decision",
    "repair_decision",
]
