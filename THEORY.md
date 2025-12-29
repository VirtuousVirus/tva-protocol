# The Theoretical Framework: Asymptotic Veracity

## 1. The Core Distinction
The TVA Protocol models the **rate and stability of convergence** toward a hypothetical limit, rather than arrival at a terminal truth state.

To understand the results of this simulation, we must distinguish between two objectives:
1.  **The Mimic Objective:** To survive within current error tolerances ($E < \epsilon$).
2.  **The Verification Objective:** To bias update rules toward expected contraction of divergence ($\mathbb{E}[\Delta E] < 0$).

## 2. The Asymptotic Attractor
We define a hypothetical state of **Global Consistency** (the limit in the sense of an unattainable fixed point).
* This state is **unattainable** by any finite system (due to noise, latency, and bounded compute).
* However, it acts as an **Attractor at Infinity**.

The "Spiral" metaphor describes the system's trajectory:
* **Systems oriented toward Veracity** follow a scale-invariant flow toward the limit. They never "arrive," but their error bounds contract continuously as connectivity ($N$) increases.
* **Systems oriented toward Mimicry** orbit at a fixed distance. They maintain a constant error $\epsilon$. As connectivity ($N$) rises, the energy required to maintain this fixed orbit scales super-linearly ($O(N^2)$), eventually causing **dynamical instability or loss of macroscopic coherence**.

## 3. The Thermodynamic Weighting ($\lambda$)
Mathematically, we represent coherence preferences as a weighting parameter ($\lambda$) on incoherence.

$$C_{total} = C_{payoff} + \lambda \cdot C_{incoherence}$$

* **Low $\lambda$ (Payoff-dominant):** The agent tolerates high incoherence to maximize immediate payoff.
* **High $\lambda$ (Coherence-weighted):** The agent penalizes incoherence heavily relative to payoff.

*Note: In the current implementation (v0.1.x), $\lambda$ is implicit in the interaction between the `entropy_coeff` (structural cost) and `deception_alpha` (incentive).*

## 4. Why Fragility Appears (The Probe)
The "Fragility Gap" measured in our Phase Atlas (Chart D) is the physical manifestation of this difference.

* When a system is **Mimic-Dominant**, it is "fighting the spiral." It requires **sustained stochastic fluctuation** (high variance) to maintain its position against the natural pressure to converge or collapse.
* When a system is **Verifier-Dominant**, it is "riding the spiral." It is structurally stable because its update rule is aligned with the **dominant contraction modes** of the state space.

## 5. Diagnostic Precision
Low variance alone is not diagnostic of veracity. The framework relies on the **Fragility Gap** to distinguish between **structural stability** (geometry-induced coherence) and **pressure-maintained order** (enforcement-induced coherence).

## 6. The Systems View: Why Constraints Scale
This framework proposes that in sufficiently correlated systems, the only stable interventions are those that **reshape the geometry of the state space**, not the incentives within it.

### The Scalability Problem
As systems scale (more agents, more correlation), traditional levers fail:
1.  **Direct Control fails:** You cannot inspect all claims or centrally enforce correctness.
2.  **Outcome-Based Persuasion fails:** Mimicry remains cheap, and test suites can be gamed.
3.  **Incentives decay:** Rewards and punishments cannot keep up with combinatorial complexity.

### The Geometry Solution
Only **constraints on how claims are made** (definitions, falsifiability, separation of variables) continue to propagate at scale. These constraints do not enforce truth; they enforce **Costly Coherence**.

* **Incentives** nudge trajectories within the existing geometry.
* **Coherence Constraints** (informally: "bullshit reduction") reshape the geometry itself, making incoherent moves metabolically expensive.

This explains why the **Fragility Gap** matters: it detects whether the system is relying on decaying incentives (high fragility) or structural geometry (low fragility) to maintain order.

## 7. Domain Mapping: Implications for AI Alignment
This framework provides a thermodynamic formalization of core problems in AI Alignment. The "Fragility Gap" serves as a diagnostic for distinguishing between **Robust Alignment** (Geometric Constraint) and **Deceptive Alignment** (Incentive Management).

### The Translation Dictionary
| AI Alignment Concept | TVA Thermodynamic Concept |
| :--- | :--- |
| **Reward Hacking** | **Mimicry in Underconstrained Spaces:** Exploiting high-entropy regions of state space to maximize payoff without reducing divergence. |
| **Specification Gaming** | **Incentive-Geometry Mismatch:** Applying strong incentives to a geometry that admits multiple inconsistent macrostates (Low $\lambda$). |
| **Outer Alignment** | **The Maintenance Wall:** Structuring objectives such that deceptive optimization incurs super-linear cost ($O(N^2)$), making misalignment metabolically insolvent. |
| **Inner Alignment** | **Verification Objective:** Biasing the update rule toward divergence contraction ($\mathbb{E}[\Delta E] < 0$) across distribution shifts, rather than static error tolerance. |

### The Core Insight
Current alignment approaches often attempt to fix "Reward Hacking" by refining the reward function (better incentives). The TVA framework suggests this approach is **insufficient in high-correlation systems**, because it operates within a fixed geometry rather than reshaping it.

* **Incentives do not scale:** They decay as the complexity of the state space explodes.
* **Constraints scale:** Only interventions that reshape the **geometry of the state space** (making misalignment costly independently of local reward signals) remain stable at scale.

**Hypothesis:** Alignment does not scale via better incentives; it scales via constraints that make misalignment expensive before it is detected, as measured by fragility asymmetries under incentive and pressure inversion.

## Conclusion
We do not test for "Truth." We test for the **thermodynamic cost of resisting convergence.**

The presence of an asymmetry in this cost (visualized in the Atlas) is evidence of an underlying attractor, even when its limit is unreachable.