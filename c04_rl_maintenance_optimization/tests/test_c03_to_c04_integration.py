import os
import sys
import json
import pytest
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from c04_rl_maintenance_optimization.src.env.compressor_maintenance_env import CompressorMaintenanceEnv
from c04_rl_maintenance_optimization.src.agents.dqn_agent import DQNAgent
from dashboard.views.c03_digital_twin_view import DigitalTwinViewController

SCHEMA_PATH = os.path.join(ROOT, "shared", "interfaces", "c03_to_c04_twin_state_contract.json")
CHECKPOINT_PATH = os.path.join(ROOT, "c04_rl_maintenance_optimization", "experiments", "checkpoints", "dqn_agent.pt")


def test_c03_contract_schema_compliance():
    """Verify that C03 Digital Twin Engine produces payloads adhering 100% to JSON contract."""
    import jsonschema

    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)

    ctrl = DigitalTwinViewController()
    ctrl.initialize_default()

    for name, sc in DigitalTwinViewController.VIVA_SCENARIOS.items():
        pkt = sc["packet"]
        ctrl.process_telemetry(pkt)
        payload = ctrl.engine.get_c04_prescriptive_payload()
        # Must not raise ValidationError
        jsonschema.validate(instance=payload, schema=schema)
        assert "operating_state" in payload
        assert "behavioral_intelligence" in payload
        assert "degradation_metrics" in payload


def test_observation_construction_from_c03_contract():
    """Verify that CompressorMaintenanceEnv maps C03 payload to 8-dim observation vector."""
    ctrl = DigitalTwinViewController()
    ctrl.initialize_default()

    for name, sc in DigitalTwinViewController.VIVA_SCENARIOS.items():
        pkt = sc["packet"]
        ctrl.process_telemetry(pkt)
        payload = ctrl.engine.get_c04_prescriptive_payload()

        obs = CompressorMaintenanceEnv.observation_from_c03_payload(payload, hours_since_last_maint=120.0)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (8,)
        assert np.all(obs >= 0.0)


def test_dqn_policy_inference_on_c03_scenarios():
    """Verify that the pre-trained DQN agent evaluates C03 states and prescribes actions."""
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip("Pretrained checkpoint not found")

    agent = DQNAgent(state_dim=8, action_dim=4)
    agent.load(CHECKPOINT_PATH)

    ctrl = DigitalTwinViewController()
    ctrl.initialize_default()

    # Step nominal
    ctrl.process_telemetry(DigitalTwinViewController.VIVA_SCENARIOS["NOMINAL_LOADED"]["packet"])
    nominal_payload = ctrl.engine.get_c04_prescriptive_payload()
    obs_nominal = CompressorMaintenanceEnv.observation_from_c03_payload(nominal_payload)
    act_nominal = agent.select_action(obs_nominal, evaluate=True)
    assert act_nominal in [0, 1]  # Should continue normal operation or minor check

    # Step bearing knock (5 frames to ensure high degradation / knock)
    for _ in range(5):
        ctrl.process_telemetry(DigitalTwinViewController.VIVA_SCENARIOS["BEARING_SPALL_DAMAGE"]["packet"])
    knock_payload = ctrl.engine.get_c04_prescriptive_payload()
    obs_knock = CompressorMaintenanceEnv.observation_from_c03_payload(knock_payload)
    # Degradation or deviation should be elevated
    assert obs_knock[0] > 0.1 or obs_knock[2] > 1.0 or obs_knock[5] > 0.0
