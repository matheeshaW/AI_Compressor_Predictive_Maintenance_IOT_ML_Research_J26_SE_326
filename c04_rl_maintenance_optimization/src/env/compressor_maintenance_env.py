"""
compressor_maintenance_env.py
=============================
Custom Gymnasium Simulation Environment for Industrial Air Compressor Maintenance Optimization.
Component 04: Reinforcement Learning-Based Maintenance Optimization
Researcher: Liyanage H.G.W.R (IT23163690)
Project: SLIIT Research Project J26-SE-326 (AIR-GUARD AI)
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .degradation_simulator import CompressorDegradationSimulator


class CompressorMaintenanceEnv(gym.Env):
    """
    Custom Gymnasium Environment modeling industrial air compressor lifecycle,
    degradation dynamics, maintenance interventions, and cost-risk trade-offs.

    Observation Space (8-dimensional continuous vector):
        0: Composite Degradation D_t in [0.0, 1.0]
        1: Normalized Remaining Useful Life RUL_t in [0.0, 1.0]
        2: Behavioral Deviation Index delta_t in [0.0, 10.0]
        3: Pressure Delivery Efficiency eta_t in [0.0, 1.0]
        4: Thermal / Current Dissipation theta_t in [0.0, 3.0]
        5: Operating Regime Index k_t in [0.0, 4.0]
        6: Anomaly Persistence Counter tau_t in [0.0, 1.0]
        7: Elapsed Cycle Age since last maintenance in [0.0, 1.0]

    Action Space (Discrete 4):
        0: OPERATE_NORMALLY (Continue) — Zero downtime, production revenue, natural wear
        1: INSPECT_AND_MINOR_SERVICE (Inspect) — 2h downtime, $200 direct cost, 40% leak wear reset
        2: SCHEDULE_MEDIUM_MAINTENANCE (Maintain) — 6h downtime, $600 direct cost, 80% valve/bearing overhaul
        3: EMERGENCY_MAJOR_MAINTENANCE (Maintain/Overhaul) — 16h downtime, $1,500 direct cost, pristine factory reset
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    # Action parameters (downtime hours, direct financial cost)
    ACTION_PARAMS = {
        0: {"name": "OPERATE_NORMALLY", "category": "Continue", "downtime_h": 0.0, "cost": 0.0},
        1: {"name": "INSPECT_AND_MINOR_SERVICE", "category": "Inspect", "downtime_h": 2.0, "cost": 200.0},
        2: {"name": "SCHEDULE_MEDIUM_MAINTENANCE", "category": "Maintain", "downtime_h": 6.0, "cost": 600.0},
        3: {"name": "EMERGENCY_MAJOR_MAINTENANCE", "category": "Overhaul", "downtime_h": 16.0, "cost": 1500.0},
    }

    # Unscheduled catastrophic breakdown parameters
    CATASTROPHIC_FAILURE_DOWNTIME_H = 48.0
    CATASTROPHIC_FAILURE_PENALTY = 5000.0

    def __init__(
        self,
        max_episode_steps: int = 300,
        step_hours: float = 4.0,
        hourly_production_value: float = 50.0,
        hourly_downtime_cost: float = 100.0,
        catastrophic_failure_penalty: float = 5000.0,
        risk_aversion_factor: float = 50.0,
        max_service_interval_hours: float = 500.0,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.step_hours = step_hours
        self.hourly_production_value = hourly_production_value
        self.hourly_downtime_cost = hourly_downtime_cost
        self.catastrophic_failure_penalty = catastrophic_failure_penalty
        self.risk_aversion_factor = risk_aversion_factor
        self.max_service_interval_hours = max_service_interval_hours
        self.render_mode = render_mode

        # Initialize physical degradation simulator
        self.simulator = CompressorDegradationSimulator(seed=seed)

        # Action space: 4 discrete maintenance and operational actions
        self.action_space = spaces.Discrete(4)

        # Observation space: 8 continuous state features
        # [D_t, RUL_t, delta_t, eta_t, theta_t, k_t, tau_t, t_maint]
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 10.0, 1.0, 3.0, 4.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Episode tracking variables
        self.current_step = 0
        self.cumulative_cost = 0.0
        self.cumulative_downtime = 0.0
        self.cumulative_production_revenue = 0.0
        self.cumulative_reward = 0.0
        self.failure_events = 0
        self.last_action = 0
        self.last_reward = 0.0

    def _get_observation(self) -> np.ndarray:
        """Extracts the 8-dimensional normalized MDP state vector."""
        telem = self.simulator.get_telemetry()
        
        comp_d = float(telem["composite_degradation"])
        norm_rul = float(telem["normalized_rul"])
        dev_idx = float(min(telem["deviation_index"], 10.0))
        press_eff = float(telem["pressure_efficiency"])
        therm_diss = float(min(telem["thermal_dissipation"], 3.0))
        regime = float(telem["operating_regime"])
        persist = float(min(telem["consecutive_anomaly_steps"] / 10.0, 1.0))
        t_maint = float(min(telem["hours_since_last_maint"] / self.max_service_interval_hours, 1.0))

        obs = np.array(
            [comp_d, norm_rul, dev_idx, press_eff, therm_diss, regime, persist, t_maint],
            dtype=np.float32
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    @classmethod
    def observation_from_c03_payload(
        cls,
        payload: Dict[str, Any],
        hours_since_last_maint: float = 0.0,
        max_service_interval_hours: float = 500.0,
    ) -> np.ndarray:
        """
        Transforms a digital twin state payload conforming to
        shared/interfaces/c03_to_c04_twin_state_contract.json into the
        canonical 8-dimensional continuous state representation s_t in S:
            0: Composite Degradation index D_t in [0.0, 1.0]
            1: Normalized RUL in [0.0, 1.0]
            2: Behavioral Deviation index delta_t in [0.0, 10.0]
            3: Operating pressure efficiency eta_t in [0.0, 1.0]
            4: Component health / thermal dissipation theta_t in [0.0, 3.0]
            5: Operating regime / fault class k_t in {0, 1, 2, 3, 4}
            6: Anomaly persistence counter tau_t in [0.0, 1.0]
            7: Elapsed cycle age since maintenance in [0.0, 1.0]
        """
        deg_metrics = payload.get("degradation_metrics", {})
        beh_intel = payload.get("behavioral_intelligence", {})
        op_state = payload.get("operating_state", {})

        comp_d = float(np.clip(deg_metrics.get("composite_degradation_score", 0.0), 0.0, 1.0))
        norm_rul = float(np.clip(1.0 - comp_d, 0.0, 1.0))
        dev_idx = float(np.clip(beh_intel.get("deviation_index", 0.0), 0.0, 10.0))
        press_eff = float(np.clip(deg_metrics.get("pressure_efficiency_ratio", 1.0), 0.0, 1.0))
        therm_diss = float(np.clip(deg_metrics.get("thermal_dissipation_factor", 1.0), 0.0, 3.0))
        regime = float(np.clip(op_state.get("cluster_id", 0), 0.0, 4.0))
        drift = bool(beh_intel.get("drift_detected", False))
        persist = 1.0 if drift else float(np.clip(beh_intel.get("persistence_counter", 0) / 5.0, 0.0, 1.0))
        t_maint = float(np.clip(hours_since_last_maint / max_service_interval_hours, 0.0, 1.0))

        obs = np.array(
            [comp_d, norm_rul, dev_idx, press_eff, therm_diss, regime, persist, t_maint],
            dtype=np.float32,
        )
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 10.0, 1.0, 3.0, 4.0, 1.0, 1.0], dtype=np.float32)
        return np.clip(obs, obs_low, obs_high)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment and underlying physical degradation simulator."""
        super().reset(seed=seed)

        if seed is not None:
            self.simulator = CompressorDegradationSimulator(seed=seed)

        initial_wear = options.get("initial_wear", None) if options else None
        self.simulator.reset(initial_wear=initial_wear)

        self.current_step = 0
        self.cumulative_cost = 0.0
        self.cumulative_downtime = 0.0
        self.cumulative_production_revenue = 0.0
        self.cumulative_reward = 0.0
        self.failure_events = 0
        self.last_action = 0
        self.last_reward = 0.0

        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes a single discrete maintenance decision step inside the environment.
        
        Calculates:
        1. Action execution / physical wear dynamics
        2. Downtime and financial expenditures
        3. Multi-objective reward R(s, a, s')
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        self.current_step += 1
        self.last_action = action

        act_meta = self.ACTION_PARAMS[action]
        action_name = act_meta["name"]
        direct_maint_cost = act_meta["cost"]
        action_downtime = act_meta["downtime_h"]

        catastrophic_failure = False
        failure_penalty = 0.0
        step_downtime = action_downtime
        failure_cause = "NORMAL"

        # Action 0: Normal continuous operation
        if action == 0:
            # Let degradation advance naturally
            catastrophic_failure, dyn_info = self.simulator.step_dynamics(hours_elapsed=self.step_hours)
            
            if catastrophic_failure:
                self.failure_events += 1
                failure_cause = dyn_info.get("failure_cause", "CATASTROPHIC_BREAKDOWN")
                step_downtime = self.CATASTROPHIC_FAILURE_DOWNTIME_H
                failure_penalty = self.catastrophic_failure_penalty
                # Emergency overhaul after catastrophic breakdown to restore machine
                self.simulator.apply_maintenance(3)
                uptime_hours = 0.0
            else:
                uptime_hours = self.step_hours

        # Actions 1, 2, 3: Prescriptive or corrective maintenance interventions
        else:
            maint_info = self.simulator.apply_maintenance(action)
            # Downtime during scheduled maintenance window
            uptime_hours = max(0.0, self.step_hours - action_downtime)

        # ----------------------------------------------------------------------
        # Multi-Objective Reward Function Formulation (as specified in docs)
        # R = R_uptime - C_direct - C_downtime - C_failure - Omega_risk
        # ----------------------------------------------------------------------
        uptime_revenue = self.hourly_production_value * uptime_hours
        downtime_loss = self.hourly_downtime_cost * step_downtime
        
        # New degradation level after action transition
        next_degradation = self.simulator.composite_degradation
        risk_penalty = self.risk_aversion_factor * (next_degradation ** 2)

        reward = float(
            uptime_revenue
            - direct_maint_cost
            - downtime_loss
            - failure_penalty
            - risk_penalty
        )

        self.last_reward = reward
        self.cumulative_reward += reward
        self.cumulative_cost += (direct_maint_cost + failure_penalty + downtime_loss)
        self.cumulative_downtime += step_downtime
        self.cumulative_production_revenue += uptime_revenue

        # Termination criteria
        terminated = bool(self.current_step >= self.max_episode_steps)
        truncated = False

        obs = self._get_observation()
        info = self._get_info()
        info.update({
            "action_taken": action,
            "action_name": action_name,
            "catastrophic_failure": catastrophic_failure,
            "failure_cause": failure_cause,
            "step_downtime_hours": step_downtime,
            "step_direct_cost": direct_maint_cost,
            "uptime_revenue": uptime_revenue,
            "risk_penalty": risk_penalty,
            "next_degradation": next_degradation,
        })

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic and KPI metrics for evaluation."""
        telem = self.simulator.get_telemetry()
        return {
            "current_step": self.current_step,
            "operating_hours": self.simulator.operating_hours,
            "cumulative_cost": self.cumulative_cost,
            "cumulative_downtime": self.cumulative_downtime,
            "cumulative_revenue": self.cumulative_production_revenue,
            "cumulative_reward": self.cumulative_reward,
            "failure_events": self.failure_events,
            "composite_degradation": self.simulator.composite_degradation,
            "pressure_psi": telem["delivered_pressure_psi"],
            "motor_current_a": telem["motor_current_a"],
            "head_temp_c": telem["head_temp_c"],
            "vibration_kurtosis": telem["vibration_kurtosis"],
            "deviation_index": telem["deviation_index"],
            "normalized_rul": telem["normalized_rul"],
            "anomaly_score": telem["anomaly_score"],
        }

    def render(self) -> Optional[str]:
        """Renders environment state to stdout or returns string buffer."""
        telem = self.simulator.get_telemetry()
        lines = [
            f"--- Step {self.current_step}/{self.max_episode_steps} | Compressor Maintenance MDP ---",
            f"Degradation: {telem['composite_degradation']:.3f} | RUL: {telem['rul_hours']:.1f}h ({telem['normalized_rul']*100:.1f}%)",
            f"Pressure: {telem['delivered_pressure_psi']:.1f} PSI | Current: {telem['motor_current_a']:.2f} A | Temp: {telem['head_temp_c']:.1f}°C",
            f"Vibration Kurtosis: {telem['vibration_kurtosis']:.2f} | Deviation: {telem['deviation_index']:.2f}",
            f"Last Action: {self.ACTION_PARAMS[self.last_action]['name']} | Last Reward: {self.last_reward:+.1f}",
            f"Total Cost: ${self.cumulative_cost:.0f} | Downtime: {self.cumulative_downtime:.1f}h | Failures: {self.failure_events}",
            "----------------------------------------------------------------",
        ]
        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output

    def close(self):
        pass
