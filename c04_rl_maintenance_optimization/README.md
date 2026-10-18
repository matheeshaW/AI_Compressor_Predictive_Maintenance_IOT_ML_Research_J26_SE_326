# Component 04: Reinforcement Learning-Based Maintenance Optimization

**Researcher:** Liyanage H.G.W.R (`IT23163690`)  
**Target Milestone:** Progress Presentation 1 (PP1) — >70% Functional Completion

---

## 1. Scope & Objectives
Component 04 moves the system from prediction to optimal prescriptive action. It formulates compressor maintenance as a Markov Decision Process (MDP) and trains an RL agent in a controlled simulation environment to select optimal maintenance actions balancing cost, downtime, and failure risk.

- **SO1:** Formulate compressor maintenance scheduling as a Markov Decision Process (MDP).
- **SO2:** Develop a custom Gymnasium simulation environment representing compressor degradation transitions.
- **SO3:** Implement conventional heuristic baselines (Run-to-Failure and Periodic Maintenance).
- **SO4:** Develop a Deep Q-Network (DQN) agent with experience replay and $\epsilon$-greedy exploration.
- **SO5:** Design a multi-objective reward function penalizing downtime and failure costs while rewarding equipment availability.
- **SO6:** Conduct comparative policy evaluation across degradation rates and cost conditions.

---

## 2. Directory Structure
```text
c04_rl_maintenance_optimization/
├── docs/
│   └── mdp_formulation.md
├── src/
│   ├── env/
│   │   ├── compressor_maintenance_env.py
│   │   └── degradation_simulator.py
│   ├── baselines/
│   │   ├── heuristic_policies.py
│   │   ├── run_to_failure_policy.py
│   │   └── periodic_maintenance_policy.py
│   ├── agents/
│   │   ├── dqn_agent.py
│   │   └── replay_buffer.py
│   └── train_dqn.py
├── experiments/
│   ├── checkpoints/
│   │   └── dqn_agent.pt
│   ├── results/
│   │   ├── comparative_policy_metrics.csv
│   │   ├── comparative_episodes_detailed.csv
│   │   ├── comparative_metrics_summary.json
│   │   ├── cost_vs_downtime_tradeoff.png
│   │   ├── cumulative_reward_comparison.png
│   │   └── policy_decision_trajectory.png
│   ├── run_comparative_experiment.py
│   └── generate_plots.py
├── notebooks/
│   └── 01_rl_policy_evaluation.ipynb
├── requirements.txt
└── tests/
    ├── conftest.py
    ├── test_baselines.py
    ├── test_comparative_experiment.py
    ├── test_compressor_env.py
    └── test_dqn_agent.py
```

---

## 3. Execution & Verification

### Run Unit Tests (35/35 Passing)
```bash
pytest c04_rl_maintenance_optimization/tests/
```

### Run 500-Episode Comparative Experiment
```bash
python c04_rl_maintenance_optimization/experiments/run_comparative_experiment.py
```

### Train DQN Agent
```bash
python c04_rl_maintenance_optimization/src/train_dqn.py
```

