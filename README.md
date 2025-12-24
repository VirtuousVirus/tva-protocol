# The Thermodynamic Veracity Alignment (TVA)

**Version:** 0.1.0-dev
**License:** MIT

The TVA is a systems architecture framework and **Agent-Based Model (ABM)** designed to investigate the thermodynamic costs of deception in multi-agent networks.

The purpose of the TVA is not to assert the inevitability of veracity, but to **make claims about veracity's stability conditions precise enough to be falsified.** It serves as an instrument to search for the specific conditions—defined by Connectivity ($N$) and Observability ($\Omega$)—under which **Veracity-Dominant Equilibrium** emerges as a metastable state within the modeled dynamics.

---

## 1. Core Principles
The model does not assume truth is inherently dominant. Instead, it models the thermodynamic friction that Deceptive Strategies must overcome to survive. The simulation focuses on two primary constraints hypothesized to define the "Veracity Phase Transition":

* **The Maintenance Wall:** A cost term modeling the energy required to maintain consistency; hypothesized to scale super-linearly ($O(n^2)$) with connectivity.
* **The Pruning Event:** A detection process where divergent agents face a rising cumulative risk of sanction as observability ($\Omega$) increases; sanctions are modeled as replacement-by-aligned (default) or bankruptcy (optional).
* **The Veracity Dividend:** The efficiency gain realized by agents who do not pay the entropy tax of maintaining a divergent reality.

*Scientific Stance:* TVA treats Veracity not as a moral imperative, but as a candidate for a **"Low-Entropy Attractor"**—a state the system naturally gravitates toward when the energy cost of deception exceeds the available resources.

---

## 2. The Simulation Engine (v0.1.0-dev)
The included `tva_simulator.py` is a fully vectorized Python engine capable of running Monte Carlo trials to stress-test these hypotheses.

### Key Features
* **Vectorized Logic:** High-performance `numpy` operations allow for rapid iteration of agent states.
* **Monte Carlo Confidence:** The engine runs multiple trials (default: 30) and aggregates the data to generate a **10th-90th percentile Uncertainty Band**.
* **Phase Sweep:** Built-in heatmap generation to visualize the transition boundary between Entropy-Dominant and Veracity-Dominant regimes.
* **Headless Ready:** Lazy imports allow execution on servers without display drivers.

### Installation
```bash
pip install numpy matplotlib
```

### How to Run
**Basic Execution (Demo Mode):**
```bash
python tva_simulator.py
```
This runs a **Phase Sweep** (Heatmap) with default parameters.

**Programmatic Use:**
```python
from tva_simulator import run_trials, save_trials_csv

# Run a single headless simulation
mean, lo, hi, runs = run_trials(
    trials=50,
    connectivity=8,
    observability=0.12,
    plot=False,
    verbose=False
)

# Export data
save_trials_csv("simulation_results.csv", mean, lo, hi, verbose=True)
```

**Research Mode: Phase Sweep**
```python
from tva_simulator import run_phase_sweep, plot_phase_sweep
import numpy as np

# Define the Grid
n_range = np.arange(1, 21, 2)         # Connectivity
omega_range = np.linspace(0, 0.5, 10) # Observability

# Run the Sweep (Computationally Intensive)
matrix = run_phase_sweep(
    n_vals=n_range, 
    omega_vals=omega_range, 
    trials=20, 
    n_agents=100
)

# Visualize
plot_phase_sweep(matrix, n_range, omega_range)
```

### Parameter Mapping
| Symbol | Code Parameter | Description |
| :--- | :--- | :--- |
| **$N$** | `connectivity` | Number of connections per agent (scales entropy cost). |
| **$\Omega$** | `observability` | Probability of detection per connection (scales risk). |
| **$\alpha$** | `deception_alpha` | Short-term payout for deceptive strategy. |
| **$E$** | `entropy_coeff` | Coefficient for the Maintenance Wall cost function. |
| **$P$** | `plasticity` | Probability an agent will switch strategies if a peer is richer. |

---

## 3. License
MIT License. Open for rigorous testing and falsification.
