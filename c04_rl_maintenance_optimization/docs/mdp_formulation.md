# Mathematical Formulation: Markov Decision Process (MDP) for Industrial Air Compressor Maintenance Optimization

**Component 04:** Reinforcement Learning-Based Prescriptive Maintenance Optimization  
**Project:** SLIIT IT4010 Research Project J26-SE-326 (AIR-GUARD AI)  
**Researcher:** Liyanage H.G.W.R (`IT23163690`)  
**Target Milestone:** Progress Presentation 1 (PP1)  

---

## 1. Research Motivation & Theoretical Scope

In industrial air compressor systems, condition monitoring systems typically yield **predictive** signals: anomaly classification alerts (Component 01 & 02) and Remaining Useful Life (RUL) estimates (Component 02). However, a prediction alone is **not an operational maintenance decision**. 

Plant managers face a fundamental economic trade-off:
1. **Premature Servicing:** Unnecessarily shutting down a machine that has useful life remaining incurs superfluous labor costs, replacement part expenses, and production interruption penalties.
2. **Delayed Servicing:** Postponing maintenance until in-service catastrophic breakdown occurs (e.g., valve shatter, seized piston, or motor thermal runaway) incurs catastrophic failure penalties that are typically 5–10× the cost of planned maintenance, alongside prolonged unplanned plant downtime.

Because physical reinforcement learning on live industrial compressors carries severe safety risks and prohibitive costs, **Component 04 establishes a rigorous Markov Decision Process (MDP) and custom Gymnasium simulation environment**. The RL agent learns an optimal prescriptive maintenance policy $\pi^*(s)$ that minimizes cumulative operational and maintenance lifecycle costs while guaranteeing equipment safety and availability.

---

## 2. Formal MDP Definition

The sequential decision-making problem is formulated as a 5-tuple:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

where:
- $\mathcal{S}$ is the continuous-discrete state space representing equipment health and operational regime.
- $\mathcal{A}$ is the discrete set of allowable maintenance and operational interventions.
- $\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0, 1]$ is the state transition probability distribution governed by compressor thermodynamics and wear mechanics.
- $\mathcal{R}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$ is the multi-objective reward function quantifying financial return, downtime penalties, and safety constraints.
- $\gamma \in [0, 1)$ is the temporal discount factor ($\gamma = 0.99$), balancing immediate intervention expenditures against long-term reliability.

---

## 3. State Space ($\mathcal{S}$)

At each discrete decision epoch $t$ (representing an operational inspection interval of $\Delta t = 4$ operating hours), the state vector $s_t \in \mathcal{S} \subset \mathbb{R}^8$ is constructed directly from the telemetry, predictive models, and Digital Twin state (`shared/interfaces/c03_to_c04_twin_state_contract.json`):

$$s_t = \begin{bmatrix} D_t, & RUL_t, & \delta_t, & \eta_t, & \theta_t, & k_t, & \tau_t, & t_{\text{maint}} \end{bmatrix}^T$$

| State Variable | Notation | Domain | Physical Description & Source |
|---|---|---|---|
| **Composite Degradation Score** | $D_t$ | $[0.0, 1.0]$ | Normalized cumulative wear index encompassing pneumatic leakage, valve seat wear, and bearing degradation. $0.0 = \text{pristine}, 1.0 = \text{total breakdown}$. |
| **Normalized Remaining Useful Life** | $RUL_t$ | $[0.0, 1.0]$ | Forecasted remaining useful life ratio $RUL_t = \frac{\tau_{\text{remain}}}{T_{\max}}$ produced by Component 02 LSTM/XGBoost models. |
| **Behavioral Deviation Index** | $\delta_t$ | $[0.0, \infty)$ | Mahalanobis / Euclidean distance from the Digital Twin (C03) learned state-specific operational envelope. |
| **Pressure Generation Efficiency** | $\eta_t$ | $[0.0, 1.0]$ | Delivery efficiency $\eta_t = \frac{P_{\text{actual}}}{P_{\text{nominal}}}$ measured by G1/4 transducer against operating load. |
| **Thermal / Current Dissipation** | $\theta_t$ | $[0.0, 2.0]$ | Normalized motor current ratio $I_{\text{RMS}} / I_{\text{rated}}$ and temperature rise factor. |
| **Operational Regime Index** | $k_t$ | $\{0, 1, 2, 3, 4\}$ | Digital Twin operational cluster: `IDLE_UNLOADED` (0), `PRESSURIZING_LOADED` (1), `HOLDING_PRESSURE` (2), `BLOWDOWN_DISCHARGE` (3), `ANOMALOUS_LEAK_REGIME` (4). |
| **Anomaly Persistence Counter** | $\tau_t$ | $[0.0, 1.0]$ | Normalized consecutive inspection windows exhibiting statistical deviation $\min(\tau / \tau_{\max}, 1.0)$. |
| **Elapsed Cycle Age** | $t_{\text{maint}}$ | $[0.0, 1.0]$ | Operating hours elapsed since last physical maintenance intervention divided by maximum scheduled service limit $T_{\text{service\_max}}$. |

---

## 4. Action Space ($\mathcal{A}$)

The action space $\mathcal{A} = \{a_0, a_1, a_2, a_3\}$ consists of 4 discrete operational and maintenance interventions reflecting established industrial plant protocols:

| Action ID | Action Name | Category | Downtime ($h$) | Direct Cost ($) | Physical Impact on Degradation ($D_t$) |
|:---:|---|---|:---:|:---:|---|
| **$a_0$** | `OPERATE_NORMALLY` | Operation | $0.0$ | $\$0$ | Machine continues running. Degradation advances according to operational load and stochastic wear dynamics. |
| **$a_1$** | `MINOR_PNEUMATIC_SERVICE` | Prescriptive | $2.0$ | $\$200$ | Clean/replace intake air filter, tighten pneumatic fittings, purge condensate trap. Reduces pneumatic leak degradation by $40\%$. |
| **$a_2$** | `MEDIUM_MECHANICAL_OVERHAUL` | Prescriptive | $6.0$ | $\$600$ | Replace compressor reed valves, rebuild cylinder head gaskets, flush lubricant lines. Reduces mechanical wear by $80\%$. |
| **$a_3$** | `MAJOR_COMPONENT_REPLACEMENT` | Corrective / Overhaul | $16.0$ | $\$1,500$ | Full motor rewind and cylinder assembly replacement. Restores equipment to pristine condition ($D_{t+1} = 0.0, RUL_{t+1} = 1.0$). |

---

## 5. State Transition Mechanics ($\mathcal{P}$)

### 5.1 Degradation Accumulation Dynamics
When the agent selects $a_0$ (`OPERATE_NORMALLY`), composite degradation accumulates following a stochastic Markovian drift process:

$$D_{t+1} = D_t + \Delta D_t$$

$$\Delta D_t = \mu_{\text{wear}}(k_t, \theta_t) + \sigma_{\text{wear}} \cdot \xi_t, \quad \xi_t \sim \mathcal{N}(0, 1)$$

where the baseline drift $\mu_{\text{wear}}$ is amplified by the operating regime $k_t$ (e.g., loaded pumping incurs $2.5\times$ higher wear rate than idle running) and thermal dissipation $\theta_t$:

$$\mu_{\text{wear}}(k_t, \theta_t) = \mu_0 \cdot \kappa(k_t) \cdot (1 + \beta \cdot \max(0, \theta_t - 1.0))$$

### 5.2 Prescriptive Maintenance Reset Dynamics
When maintenance actions $a \in \{a_1, a_2, a_3\}$ are executed, the degradation state transitions according to restoration efficiency factors:

$$D_{t+1} = \max\left(0.0, D_t \cdot (1 - \rho_a) + \epsilon_{\text{res}}\right)$$

where:
- $\rho_{a_1} = 0.40$ (Minor pneumatic filter clean)
- $\rho_{a_2} = 0.80$ (Medium mechanical valve overhaul)
- $\rho_{a_3} = 1.00$ (Major assembly replacement $\rightarrow D_{t+1} = 0.0$)
- $\epsilon_{\text{res}} \sim \mathcal{N}(0, 0.01)$ represents minor procedural variance in industrial servicing.

### 5.3 Catastrophic Failure Hazard Function
If composite degradation reaches $D_t \ge 1.0$, or if an unmitigated critical mechanical knock / thermal runaway triggers an unexpected breakdown before scheduled intervention, the machine transitions into the **Failed absorbing state**:

$$\mathbb{P}(\text{Failure} \mid D_t, a_0) = \begin{cases} 
1.0 & \text{if } D_t \ge 1.0 \\
1 - \exp\left(-\lambda_0 \cdot \exp(\alpha_{\text{fail}} \cdot D_t)\right) & \text{if } D_t < 1.0 
\end{cases}$$

A catastrophic breakdown incurs:
- Unplanned Emergency Downtime: $48.0\text{ hours}$
- Catastrophic Breakdown Penalty: $C_{\text{failure}} = \$5,000$ (emergency contractor dispatch, collateral impeller damage, urgent shipment of replacement parts).

---

## 6. Multi-Objective Reward Function ($\mathcal{R}$)

The reward function mathematically encodes the multi-objective optimization problem: maximizing production uptime while minimizing direct maintenance and downtime costs:

$$R(s_t, a_t, s_{t+1}) = R_{\text{uptime}} - C_{\text{direct}}(a_t) - C_{\text{downtime}}(a_t) - \mathbb{I}_{\text{fail}} \cdot C_{\text{failure}} - \Omega_{\text{risk}}(D_{t+1})$$

where:
1. **Production Uptime Revenue:**
   $$R_{\text{uptime}} = V_{\text{prod}} \cdot (\Delta t - h_{\text{down}}(a_t))$$
   where $V_{\text{prod}} = \$50/\text{hr}$ is the economic value generated by continuous pressurized air delivery.
2. **Direct Action Maintenance Cost:**
   $$C_{\text{direct}}(a_t) \in \{\$0, \$200, \$600, \$1,500\}$$
3. **Operational Downtime Penalty:**
   $$C_{\text{downtime}}(a_t) = C_{\text{hourly\_down}} \cdot h_{\text{down}}(a_t)$$
   where $C_{\text{hourly\_down}} = \$100/\text{hr}$ accounts for downstream production line idle costs during scheduled maintenance windows.
4. **Catastrophic Failure Penalty:**
   $$\mathbb{I}_{\text{fail}} \cdot C_{\text{failure}} = \begin{cases} \$5,000 & \text{if catastrophic failure occurs} \\ \$0 & \text{otherwise} \end{cases}$$
5. **Degradation Risk Aversion Term:**
   $$\Omega_{\text{risk}}(D_{t+1}) = \omega_{\text{risk}} \cdot (D_{t+1})^2$$
   penalizing the agent for allowing equipment to operate under dangerous degradation levels ($\omega_{\text{risk}} = 50.0$).

---

## 7. Deep Q-Network (DQN) Solution Architecture

To solve this continuous-state MDP without discretization artifacts, we employ **Deep Q-Networks (DQN)** with:
1. **Q-Network Approximation:** $Q(s, a; \theta) \approx Q^*(s, a)$ parameterized by a multi-layer perceptron (MLP) with Layer Normalization and LeakyReLU activations.
2. **Experience Replay Buffer $\mathcal{D}$:** Uniform random sampling of transitions $(s_t, a_t, r_t, s_{t+1}, d_t)$ to break temporal autocorrelation in sequential telemetry.
3. **Target Network $\theta^-$:** Soft or periodic Polyak target updates $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$ to eliminate moving-target instability.
4. **Bellman Optimality Loss:**
   $$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \left( r + \gamma (1-d) \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
5. **$\epsilon$-Greedy Exploration:** Decaying from $\epsilon_{\text{start}} = 1.0$ down to $\epsilon_{\text{end}} = 0.05$ with an exponential decay factor.

---

## 8. Benchmark Policies for Empirical Comparison

To validate the research contribution in accordance with the SLIIT rubric (LO2, LO4, LO5), DQN is evaluated against standard industrial baselines over 500 simulated production episodes:
1. **Run-to-Failure (RTF):** Reactive maintenance policy ($a_t = a_0$ until $D_t \ge 1.0$ or catastrophic failure, then execute $a_3$).
2. **Periodic Maintenance (PM):** Traditional preventive maintenance (execute minor service $a_1$ every 50 hours, major overhaul $a_3$ every 200 hours, regardless of true sensor state).
3. **Threshold-Based Condition Maintenance (CBM):** Heuristic rule engine triggering maintenance when composite degradation $D_t > 0.70$.
4. **DQN Prescriptive Policy ($\pi^*$):** Reinforcement learning prescriptive policy selecting actions dynamically by maximizing expected discounted lifecycle return $Q^*(s, a)$.
