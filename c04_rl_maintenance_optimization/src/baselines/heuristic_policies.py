"""
heuristic_policies.py
=====================
Conventional Heuristic Baseline Maintenance Policies for Industrial Air Compressors.
Component 04: Reinforcement Learning-Based Maintenance Optimization
Researcher: Liyanage H.G.W.R (IT23163690)
Project: SLIIT Research Project J26-SE-326 (AIR-GUARD AI)

Implements:
1. RunToFailurePolicy (RTF): Pure reactive maintenance. Continues normal operation
   until catastrophic failure occurs or Remaining Useful Life (RUL) depletes to 0.0,
   triggering Emergency Major Maintenance (Action 3).
2. PeriodicMaintenancePolicy (PM): Time/step-based preventive maintenance. Executes
   normal operation until a scheduled interval is reached, triggering Medium
   Mechanical Overhaul (Action 2) and resetting its internal timer/counter.
3. ConditionBasedMaintenancePolicy (CBM): Threshold-based predictive maintenance.
   Triggers proactive maintenance when composite degradation or RUL crosses defined limits.
"""

from typing import Dict, Any, Optional, Union
import numpy as np


# Discrete Action Definitions (consistent with CompressorMaintenanceEnv)
ACTION_OPERATE_NORMALLY = 0
ACTION_INSPECT_AND_MINOR_SERVICE = 1
ACTION_SCHEDULE_MEDIUM_MAINTENANCE = 2
ACTION_EMERGENCY_MAJOR_MAINTENANCE = 3


class BasePolicy:
    """Abstract base interface for maintenance decision policies."""

    def select_action(
        self,
        state: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> int:
        raise NotImplementedError("Subclasses must implement select_action()")

    def act(
        self,
        state: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> int:
        return self.select_action(state, info, **kwargs)

    def __call__(
        self,
        state: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> int:
        return self.select_action(state, info, **kwargs)

    def reset(self) -> None:
        """Resets any internal policy state or counters."""
        pass


class RunToFailurePolicy(BasePolicy):
    """
    Run-to-Failure (RTF) Heuristic Maintenance Policy.

    A purely reactive baseline representing conventional unmonitored equipment operation.
    Always returns action 0 (OPERATE_NORMALLY), unless the equipment state indicates
    a catastrophic failure or Remaining Useful Life RUL <= 0.0, at which point it
    returns action 3 (EMERGENCY_MAJOR_MAINTENANCE).

    Parameters:
        rul_threshold: RUL limit below which emergency overhaul is mandatory (default 0.0).
        degradation_threshold: Composite degradation score indicating total breakdown (default 1.0).
    """

    def __init__(
        self,
        rul_threshold: float = 0.0,
        degradation_threshold: float = 1.0,
    ):
        self.rul_threshold = float(rul_threshold)
        self.degradation_threshold = float(degradation_threshold)
        self.total_emergency_repairs: int = 0
        self.total_steps: int = 0

    def select_action(
        self,
        state: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> int:
        """
        Evaluates state and returns maintenance action.

        Args:
            state: Continuous MDP observation vector (8-dim), dictionary, or scalar RUL.
            info: Optional diagnostics dictionary returned by environment step.

        Returns:
            int: 0 (OPERATE_NORMALLY) or 3 (EMERGENCY_MAJOR_MAINTENANCE).
        """
        self.total_steps += 1
        is_failed = False

        # 1. Check environment diagnostic info dictionary if available
        if info is not None:
            if info.get("catastrophic_failure", False):
                is_failed = True
            elif info.get("failure_cause", "NORMAL") not in ("NORMAL", None, ""):
                is_failed = True
            elif "normalized_rul" in info and float(info["normalized_rul"]) <= self.rul_threshold:
                is_failed = True
            elif "rul_hours" in info and float(info["rul_hours"]) <= 0.0:
                is_failed = True

        # 2. Check state representation
        if not is_failed and state is not None:
            # Case A: Dictionary state representation
            if isinstance(state, dict):
                if state.get("catastrophic_failure", False) or state.get("is_failed", False):
                    is_failed = True
                else:
                    for rul_key in ("normalized_rul", "rul", "RUL", "rul_hours"):
                        if rul_key in state:
                            if float(state[rul_key]) <= self.rul_threshold:
                                is_failed = True
                            break
                    if not is_failed:
                        for deg_key in ("composite_degradation", "degradation", "wear"):
                            if deg_key in state:
                                if float(state[deg_key]) >= self.degradation_threshold:
                                    is_failed = True
                                break

            # Case B: Scalar numeric state (directly passing RUL or health score)
            elif isinstance(state, (int, float, np.floating, np.integer)):
                if float(state) <= self.rul_threshold:
                    is_failed = True

            # Case C: Array-like observation vector (e.g. 8-dim MDP state from Gymnasium)
            # [0: comp_degradation, 1: normalized_rul, 2: dev_idx, 3: press_eff, ...]
            else:
                try:
                    arr = np.asarray(state, dtype=np.float32).flatten()
                    if arr.size >= 2:
                        comp_deg = float(arr[0])
                        norm_rul = float(arr[1])
                        if norm_rul <= self.rul_threshold or comp_deg >= self.degradation_threshold:
                            is_failed = True
                    elif arr.size == 1:
                        if float(arr[0]) <= self.rul_threshold:
                            is_failed = True
                except (ValueError, TypeError):
                    pass

        if is_failed:
            self.total_emergency_repairs += 1
            return ACTION_EMERGENCY_MAJOR_MAINTENANCE

        return ACTION_OPERATE_NORMALLY

    def reset(self) -> None:
        """Resets policy episode tracking statistics."""
        self.total_emergency_repairs = 0
        self.total_steps = 0


class PeriodicMaintenancePolicy(BasePolicy):
    """
    Periodic Maintenance (PM) Heuristic Policy.

    A traditional scheduled preventive maintenance policy operating independently
    of sensor feedback. Returns action 0 (OPERATE_NORMALLY) during regular operation,
    and returns action 2 (SCHEDULE_MEDIUM_MAINTENANCE) exactly when the scheduled
    maintenance interval (in steps or operating hours) is reached, immediately
    resetting its internal counter.

    Parameters:
        maintenance_interval: Steps or hours elapsed between scheduled services.
        step_hours: Operating hours per discrete environment step (default 4.0 if tracking hours).
        unit: 'steps' (default) or 'hours' defining the counter measurement mode.
        mode: Alias for unit ('steps' or 'hours').
    """

    def __init__(
        self,
        maintenance_interval: Union[int, float],
        step_hours: Optional[float] = None,
        unit: str = "steps",
        mode: Optional[str] = None,
    ):
        if maintenance_interval <= 0:
            raise ValueError(f"maintenance_interval must be positive, got {maintenance_interval}")

        self.maintenance_interval = float(maintenance_interval)
        self.step_hours = float(step_hours) if step_hours is not None else 4.0
        self.unit = (mode or unit).lower()

        if self.unit not in ("steps", "hours"):
            raise ValueError(f"unit/mode must be 'steps' or 'hours', got '{self.unit}'")

        self.internal_counter: float = 0.0
        self.total_scheduled_services: int = 0
        self.total_steps: int = 0

    @property
    def counter(self) -> Union[int, float]:
        """Current internal counter value."""
        if self.unit == "steps":
            return int(round(self.internal_counter))
        return float(self.internal_counter)

    @counter.setter
    def counter(self, val: Union[int, float]) -> None:
        self.internal_counter = float(val)

    def select_action(
        self,
        state: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None,
        hours_elapsed: Optional[float] = None,
        **kwargs: Any,
    ) -> int:
        """
        Advances the internal counter and determines the maintenance action.

        Args:
            state: Optional observation state (ignored by time-based periodic maintenance).
            info: Optional diagnostics dictionary.
            hours_elapsed: Operating hours elapsed during this step (used in 'hours' mode).

        Returns:
            int: 2 (SCHEDULE_MEDIUM_MAINTENANCE) exactly when the interval is reached,
                 0 (OPERATE_NORMALLY) otherwise.
        """
        self.total_steps += 1

        # Advance internal counter
        if self.unit == "hours":
            delta_h = hours_elapsed if hours_elapsed is not None else self.step_hours
            self.internal_counter += delta_h
        else:
            self.internal_counter += 1.0

        # Trigger scheduled maintenance when interval is reached
        if self.internal_counter >= self.maintenance_interval:
            self.internal_counter = 0.0
            self.total_scheduled_services += 1
            return ACTION_SCHEDULE_MEDIUM_MAINTENANCE

        return ACTION_OPERATE_NORMALLY

    def reset(self) -> None:
        """Resets internal interval counter and statistics."""
        self.internal_counter = 0.0
        self.total_scheduled_services = 0
        self.total_steps = 0


class ConditionBasedMaintenancePolicy(BasePolicy):
    """
    Condition-Based Maintenance (CBM) Heuristic Policy.

    A predictive threshold-based baseline triggering maintenance interventions
    when physical degradation or RUL crosses pre-configured safety thresholds.

    Parameters:
        degradation_threshold: Composite degradation trigger limit (default 0.70).
        rul_threshold: Normalized RUL lower trigger limit (default 0.20).
        maintenance_action: Action to execute when threshold is breached (default 2).
    """

    def __init__(
        self,
        degradation_threshold: float = 0.70,
        rul_threshold: float = 0.20,
        maintenance_action: int = ACTION_SCHEDULE_MEDIUM_MAINTENANCE,
    ):
        self.degradation_threshold = float(degradation_threshold)
        self.rul_threshold = float(rul_threshold)
        self.maintenance_action = maintenance_action
        self.total_interventions: int = 0

    def select_action(
        self,
        state: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> int:
        trigger = False

        if state is not None:
            if isinstance(state, dict):
                comp_deg = state.get("composite_degradation", 0.0)
                norm_rul = state.get("normalized_rul", 1.0)
                if comp_deg >= self.degradation_threshold or norm_rul <= self.rul_threshold:
                    trigger = True
            else:
                try:
                    arr = np.asarray(state, dtype=np.float32).flatten()
                    if arr.size >= 2:
                        comp_deg = float(arr[0])
                        norm_rul = float(arr[1])
                        if comp_deg >= self.degradation_threshold or norm_rul <= self.rul_threshold:
                            trigger = True
                except (ValueError, TypeError):
                    pass

        if trigger:
            self.total_interventions += 1
            return self.maintenance_action

        return ACTION_OPERATE_NORMALLY

    def reset(self) -> None:
        self.total_interventions = 0
