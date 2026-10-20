"""
run_to_failure_policy.py
========================
Run-to-Failure (RTF) Heuristic Baseline Policy.
"""

from .heuristic_policies import (
    RunToFailurePolicy,
    ACTION_OPERATE_NORMALLY,
    ACTION_EMERGENCY_MAJOR_MAINTENANCE,
)

__all__ = [
    "RunToFailurePolicy",
    "ACTION_OPERATE_NORMALLY",
    "ACTION_EMERGENCY_MAJOR_MAINTENANCE",
]
