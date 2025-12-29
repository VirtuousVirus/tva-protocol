# Thermodynamic Veracity Alignment (TVA)
**A probe suite for mapping the feasibility conditions of verification-seeking protocols.**

**Version:** 0.1.5-stable
**License:** MIT

## Abstract
This project investigates a specific thermodynamic hypothesis: that in multi-agent systems with high connectivity and observability, the effective cost of maintaining divergent states appears to scale super-linearly (e.g., $O(N^2)$), while **consistent strategies avoid that scaling burden**.

If this asymmetry holds, it implies the existence of a "Veracity Attractor"—an operational region in state space where minimizing divergence is energetically dominant over mimicry.

## Project Status: The Probe Phase
Currently, this repository implements the **Modeling Framework** and **Falsification Probes** required to detect this attractor. It explicitly separates the "Maintenance Wall" (structural cost) from "Social Pressure" (enforcement) to isolate the thermodynamic signature of veracity.

> **Note:** This repository is currently a *diagnostic instrument*, not a *prescriptive standard*. It maps the conditions under which a verification protocol *could* exist. The construction of the Protocol itself—the specific cryptographic or behavioral update rules that exploit this gradient—is the active open problem defined by these findings.

---

## Part I: The Theoretical Premises

For a deeper analysis of the asymptotic convergence model, see [THEORY.md](THEORY.md).

### 1. Lemma: The Conditional Scaling of Deception
**Premise:** In a network characterized by Persistent Identity and Low-Cost State Correlation, the effective cost of Deception appears to scale super-linearly.

* **Broadcast Deception ($$O(1)$$):** Low cost, but Fragile. A single state leak to any observer invalidates the state for all observers.
* **Contextual Deception ($$O(N^2)$$):** Robust, but Expensive. Requires maintaining distinct, non-colliding state ledgers for each observer.

**Conclusion:** As network transparency increases (marginal cross-validation cost $\to$ low), Contextual Deception becomes computationally insolvent. The TVA Simulator explicitly models this via the "Maintenance Wall."

### 2. The Truth Investment Curve (Risk-Adjusted)
Veracity is modeled as a low-maintenance strategy with a distinct risk profile.
* **Cost Profile:** High Initialization + Near-Zero Maintenance.
* **The Exposure Constraint:** Veracity is strictly dominant only if the tail risk of exploitation (adversarial free-riding on honest signals) is lower than the cumulative maintenance cost of inconsistent ledgers.

### 3. Falsification Targets
To distinguish this framework from ideology, we define the specific conditions under which the hypothesis fails. The TVA model is considered **falsified** if large regions of parameter space exist where:

* **(a)** Deception remains dominant or metastable under conditions of High Connectivity ($N > 20$) and High Observability ($\Omega > 0.3$), as measured by final alignment scores (mean aligned fraction over the last 20% of timesteps).
* **(b)** The "Maintenance Wall" fails to offset the short-term Deception Alpha over long time horizons (divergence does not collapse).
* **(c)** The **Fragility Gap** (defined in [INTERPRETATION.md](INTERPRETATION.md)) remains near zero within confidence bands when **thermodynamic roles are swapped**, implying the system is purely reward-driven rather than structurally constrained.

---

## Part II: The Simulation Engine

The included `tva_simulator.py` is the computational engine used to stress-test these claims. It supports orthogonal decomposition of **Thermodynamic Role Swaps** vs. **Social Field Inversion** to identify symmetry breaking.

### Key Features
* **Vectorized Logic:** High-performance `numpy` operations allow for rapid iteration of agent states.
* **Orthogonal Probes:** Independent control over thermodynamic load distribution (Role Swap) and enforcement targets (Pruning) to separate structural forces from social forces.
* **Thermodynamic Metrics:** Measures not just mean outcomes, but the "Fragility Gap" (trial-dispersion asymmetry) required to maintain them.
* **Headless Ready:** Optional plotting imports allow execution on servers without display drivers.

### Installation

    pip install -r requirements.txt

### Usage

**1. Run the "Symmetry Probe" (The Atlas)**
This generates the Decomposed Atlas (Heatmaps) that visualizes the Fragility Gap.

    python generate_decomposed_atlas.py

*Output:* `tva_atlas_decomposed.png` (See [INTERPRETATION.md](INTERPRETATION.md) for how to read this chart).

**2. Programmatic Experimentation**

    from tva_simulator import run_trials

    # Run a specific scenario: High Connectivity, Thermodynamic Role Swap
    mean, lo, hi, runs, width = run_trials(
        trials=50,
        connectivity=25,
        observability=0.2,
        invert_payoffs=True, # Test structural resistance via Role Swap
        plot=True
    )

---

## License

Distributed under the MIT License. Open for rigorous testing and falsification.
