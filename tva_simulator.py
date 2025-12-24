import numpy as np

# TVA Engine v0.1.0-dev
# The Thermodynamic Veracity Alignment (TVA) Protocol
# Repository: tva-protocol

ALIGNED = 1
DIVERGENT = 0

def run_tva_simulation(
    n_agents=100,
    steps=100,
    connectivity=5,
    observability=0.1,
    entropy_coeff=0.05,      # Scales the Maintenance Wall
    deception_alpha=2.0,     # Short-term gain for divergent strategy
    veracity_dividend=1.0,   # Steady gain for aligned strategy
    prune_to_alignment=True, # If caught: forced compliance/replacement
    replacement_wealth=10.0, # Wealth of the replacement agent
    plasticity=0.1,          # Probability to copy a richer peer
    clip_wealth_at_zero=True,# Prevent negative wealth artifacts
    avoid_self_compare=True, # Prevent agents from comparing to themselves
    plot=True,               # Toggle plotting for single runs
    seed=None,
):
    # --- Input Validation ---
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if n_agents < 2:
        raise ValueError("n_agents must be >= 2")
    if connectivity < 1:
        raise ValueError("connectivity must be >= 1")
    if connectivity > n_agents - 1:
        raise ValueError(f"connectivity ({connectivity}) cannot exceed n_agents - 1 ({n_agents - 1})")
    if not (0.0 <= observability <= 1.0):
        raise ValueError("observability must be in [0.0, 1.0]")
    if not (0.0 <= plasticity <= 1.0):
        raise ValueError("plasticity must be in [0.0, 1.0]")

    rng = np.random.default_rng(seed)

    # Explicit 50/50 split initialization
    agents = rng.choice([ALIGNED, DIVERGENT], size=n_agents, p=[0.5, 0.5])
    wealth = np.ones(n_agents) * 10.0

    # The Physics: Maintenance Wall (Energy) & Ruin (Probability)
    entropy_cost = (connectivity ** 2) * entropy_coeff
    pruning_prob = 1.0 - (1.0 - observability) ** connectivity

    history_alignment = np.empty(steps, dtype=float)

    for t in range(steps):
        is_div = (agents == DIVERGENT)
        is_aln = ~is_div

        # --- Payoffs ---
        wealth[is_aln] += veracity_dividend
        wealth[is_div] += (deception_alpha - entropy_cost)

        # --- Pruning (The Audit) ---
        caught = is_div & (rng.random(n_agents) < pruning_prob)
        if np.any(caught):
            if prune_to_alignment:
                # MODELING CHOICE: Replacement
                agents[caught] = ALIGNED
                wealth[caught] = replacement_wealth
            else:
                # MODELING CHOICE: Ruin
                wealth[caught] = 0.0

        # --- Wealth Clipping ---
        if clip_wealth_at_zero:
            np.maximum(wealth, 0.0, out=wealth)

        # --- Evolution (Pairwise Imitation) ---
        if avoid_self_compare:
            # Vectorized non-self selection: (i + random(1, N)) % N
            shift = rng.integers(1, n_agents, size=n_agents)
            peer_idx = (np.arange(n_agents) + shift) % n_agents
        else:
            peer_idx = rng.integers(0, n_agents, size=n_agents)

        richer_peer = wealth[peer_idx] > wealth
        imitate = richer_peer & (rng.random(n_agents) < plasticity)
        
        # Apply strategy update
        agents[imitate] = agents[peer_idx[imitate]]

        # Record State
        history_alignment[t] = np.mean(agents)

    # Visualization (Single Run)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(history_alignment, label="Alignment (Veracity)", color="green", linewidth=2)
        plt.title(f"TVA Single Run: Ω={observability}, N={connectivity}")
        plt.xlabel("Time Steps")
        plt.ylabel("% Aligned Agents")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    return history_alignment

def run_trials(trials=30, base_seed=42, plot=True, verbose=True, **kwargs):
    """
    Runs multiple simulations and returns stats.
    Returns: (mean, lo, hi, raw_runs)
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if base_seed is None:
        base_seed = 0
    # Redundant check for better UX (catches steps=0 before inner loop)
    if "steps" in kwargs and kwargs["steps"] < 1:
        raise ValueError("steps must be >= 1")

    if verbose:
        print(f"Running {trials} Monte Carlo trials...")
    
    # Safe kwarg handling
    trial_kwargs = dict(kwargs)
    trial_kwargs["plot"] = False
    
    runs = np.vstack([
        run_tva_simulation(seed=base_seed + i, **trial_kwargs) 
        for i in range(trials)
    ])
    
    mean = runs.mean(axis=0)
    lo = np.quantile(runs, 0.1, axis=0)
    hi = np.quantile(runs, 0.9, axis=0)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        x = np.arange(mean.size)
        plt.fill_between(x, lo, hi, color="green", alpha=0.2, label="10–90% band")
        plt.plot(x, mean, color="green", linewidth=2, label="Mean Alignment")
        plt.title(f"TVA Aggregate: Ω={trial_kwargs.get('observability', 0.1)}, N={trial_kwargs.get('connectivity', 5)}")
        plt.xlabel("Time Steps")
        plt.ylabel("% Aligned Agents")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        plt.show()
    
    return mean, lo, hi, runs

def run_phase_sweep(n_vals, omega_vals, trials=20, steps=100, verbose=True, **kwargs):
    """
    Runs a grid search over Connectivity (N) and Observability (Ω).
    Returns: heatmap_matrix (shape: len(omega_vals) x len(n_vals))
    """
    # --- Sweep Validation ---
    n_vals_arr = np.array(n_vals)
    omega_vals_arr = np.array(omega_vals)
    
    if np.any(n_vals_arr < 1):
        raise ValueError("All n_vals must be >= 1")
    if np.any((omega_vals_arr < 0.0) | (omega_vals_arr > 1.0)):
        raise ValueError("All omega_vals must be in [0.0, 1.0]")
        
    n_agents = kwargs.get("n_agents", 100)
    if np.max(n_vals_arr) > n_agents - 1:
        raise ValueError(f"Max connectivity ({np.max(n_vals_arr)}) cannot exceed n_agents-1 ({n_agents-1}). Increase n_agents.")

    heatmap = np.zeros((len(omega_vals), len(n_vals)))
    
    if verbose:
        print(f"Starting Phase Sweep: {len(n_vals)}x{len(omega_vals)} grid, {trials} trials/cell.")

    for i, omega in enumerate(omega_vals):
        for j, n in enumerate(n_vals):
            # Run trials headless
            mean, _, _, _ = run_trials(
                trials=trials, 
                connectivity=int(n), 
                observability=omega, 
                steps=steps, 
                plot=False, 
                verbose=False,
                **kwargs
            )
            # Metric: Average alignment over the last 20% of the run
            final_score = mean[int(steps*0.8):].mean()
            heatmap[i, j] = final_score
            
        if verbose:
            print(f"Finished row {i+1}/{len(omega_vals)} (Ω={omega:.2f})")
            
    return heatmap

def plot_phase_sweep(heatmap, n_vals, omega_vals):
    """
    Visualizes the Phase Diagram Heatmap using index-space plotting for precise ticks.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    # Plot in index space (0..len-1) to ensure rectangular cells match ticks perfectly
    plt.imshow(heatmap, origin='lower', aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    plt.colorbar(label="Final Alignment Score (Green=Veracity, Red=Entropy)")
    
    # Map index ticks to actual values
    # X-Axis: Connectivity
    plt.xticks(np.arange(len(n_vals)), n_vals)
    
    # Y-Axis: Observability (Round to avoid ugly floats)
    # If too many y-ticks, thin them out (display every 2nd label)
    if len(omega_vals) > 10:
        y_indices = np.arange(0, len(omega_vals), 2)
        y_labels = np.round(omega_vals[::2], 3)
    else:
        y_indices = np.arange(len(omega_vals))
        y_labels = np.round(omega_vals, 3)
        
    plt.yticks(y_indices, y_labels)

    plt.xlabel("Connectivity ($N$)")
    plt.ylabel("Observability ($\Omega$)")
    plt.title("TVA Phase Diagram: The Veracity Transition (v0.1.0-dev)")
    plt.grid(False)
    plt.show()

def save_trials_csv(path, mean, lo, hi, verbose=False):
    data = np.column_stack([mean, lo, hi])
    header = "mean_alignment,lo_p10,hi_p90"
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    if verbose:
        print(f"Saved simulation data to {path}")

if __name__ == "__main__":
    # Example: Run a Demo Phase Sweep
    print("Running Demo Phase Sweep (v0.1.0-dev)...")
    
    # Connectivity: 1 to 20 (Must be <= n_agents-1)
    n_range = np.arange(1, 16, 2)       
    
    # Observability: 0.0 to 0.4
    omega_range = np.linspace(0, 0.4, 10) 
    
    # Passing n_agents=100 explicitly to be safe
    matrix = run_phase_sweep(n_range, omega_range, trials=10, steps=100, n_agents=100)
    plot_phase_sweep(matrix, n_range, omega_range)
