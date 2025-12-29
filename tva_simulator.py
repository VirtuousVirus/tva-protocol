import numpy as np

# TVA Engine v0.1.5-stable
# The Thermodynamic Veracity Alignment (TVA) Probe Suite
# Repository: tva-protocol

ALIGNED = 1
DIVERGENT = 0

def calculate_entropy_coefficient(break_even_n, alpha, dividend):
    if break_even_n <= 0: return 0.0
    return (alpha - dividend) / (break_even_n ** 2)

def run_tva_simulation(
    n_agents=100,
    steps=100,
    connectivity=5,
    observability=0.1,
    break_even_connectivity=20, 
    entropy_coeff=None,         
    deception_alpha=2.0,     
    veracity_dividend=1.0,   
    enforce_pruning=True,    # If False: "Sanction Mode" (wealth destroyed, state kept)
    replacement_wealth=10.0, 
    plasticity=0.1,          
    clip_wealth_at_zero=True,
    avoid_self_compare=True, 
    invert_payoffs=False,    # PROBE A: Thermodynamic Role Swap
    invert_pruning=False,    # PROBE B: Social Field Inversion
    plot=False,              
    seed=None,
):
    """
    Runs a single simulation of the TVA process.
    
    Note on Probes:
      - invert_payoffs swaps which label bears the structural cost (entropy); 
        it does NOT change the pruning target unless invert_pruning is also set.
      - invert_pruning swaps the social pressure target (who gets audited).
      - Setting both creates a fully inverted world; setting one creates conflict.
    """
    
    # --- Input Validation ---
    if steps < 1: raise ValueError("steps >= 1")
    if n_agents < 2: raise ValueError("n_agents >= 2")
    if connectivity < 1: raise ValueError("connectivity >= 1")
    if connectivity > n_agents - 1: raise ValueError("connectivity limit")
    if not (0.0 <= observability <= 1.0): raise ValueError("observability [0,1]")

    # --- Physics Calibration ---
    if entropy_coeff is None:
        entropy_coeff = calculate_entropy_coefficient(
            break_even_connectivity, deception_alpha, veracity_dividend
        )

    rng = np.random.default_rng(seed)
    agents = rng.choice([ALIGNED, DIVERGENT], size=n_agents, p=[0.5, 0.5])
    wealth = np.ones(n_agents) * 10.0

    entropy_cost = (connectivity ** 2) * entropy_coeff
    pruning_prob = 1.0 - (1.0 - observability) ** connectivity

    history_alignment = np.empty(steps, dtype=float)

    for t in range(steps):
        is_div = (agents == DIVERGENT)
        is_aln = ~is_div

        # --- 1. Payoffs (Thermodynamic Role Swap) ---
        if invert_payoffs:
            # SWAP: Aligned bears the N^2 cost; Divergent gets dividend.
            wealth[is_aln] += (deception_alpha - entropy_cost)
            wealth[is_div] += veracity_dividend
        else:
            # STANDARD: Divergent bears N^2 cost; Aligned gets dividend.
            wealth[is_aln] += veracity_dividend
            wealth[is_div] += (deception_alpha - entropy_cost)

        # --- 2. Pruning (Social Field Direction) ---
        if invert_pruning:
            # INVERTED: System audits ALIGNED agents.
            caught = is_aln & (rng.random(n_agents) < pruning_prob)
            forced_state = DIVERGENT
        else:
            # STANDARD: System audits DIVERGENT agents.
            caught = is_div & (rng.random(n_agents) < pruning_prob)
            forced_state = ALIGNED

        if np.any(caught):
            if enforce_pruning:
                # Replacement Mode: Force state change + reset wealth
                agents[caught] = forced_state
                wealth[caught] = replacement_wealth
            else:
                # Sanction Mode: Destroy wealth, keep state (Persistent Dissident)
                wealth[caught] = 0.0

        # --- 3. Evolution (Imitation) ---
        if clip_wealth_at_zero:
            np.maximum(wealth, 0.0, out=wealth)

        if avoid_self_compare:
            shift = rng.integers(1, n_agents, size=n_agents)
            peer_idx = (np.arange(n_agents) + shift) % n_agents
        else:
            peer_idx = rng.integers(0, n_agents, size=n_agents)

        richer_peer = wealth[peer_idx] > wealth
        imitate = richer_peer & (rng.random(n_agents) < plasticity)
        
        agents[imitate] = agents[peer_idx[imitate]]
        history_alignment[t] = np.mean(agents)

    # --- Visualization (Single Run) ---
    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.plot(history_alignment, label="Alignment", color="green", linewidth=2)
            plt.title(f"TVA Single Run: Ω={observability}, N={connectivity}, Steps={steps}")
            plt.xlabel("Steps")
            plt.ylabel("Alignment %")
            plt.ylim(-0.05, 1.05)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()
        except ImportError:
            print("Warning: matplotlib not installed, skipping plot.")

    return history_alignment

def run_trials(trials=30, base_seed=42, plot=False, verbose=False, **kwargs):
    """
    Runs multiple simulations.
    Returns: (mean, lo, hi, runs, uncertainty_width)
    """
    if trials < 1: raise ValueError("trials >= 1")
    if base_seed is None: base_seed = 0
    
    # Clean kwargs to ensure strict control over plotting
    trial_kwargs = dict(kwargs)
    trial_kwargs.pop("plot", None)
    trial_kwargs["plot"] = False
    
    if verbose:
        print(f"Running {trials} trials...")

    runs = np.vstack([
        run_tva_simulation(seed=base_seed + i, **trial_kwargs) 
        for i in range(trials)
    ])
    
    mean = runs.mean(axis=0)
    lo = np.quantile(runs, 0.1, axis=0)
    hi = np.quantile(runs, 0.9, axis=0)
    uncertainty_width = hi - lo

    # --- Visualization (Aggregate) ---
    if plot:
        try:
            import matplotlib.pyplot as plt
            # Fetch defaults for cleaner title if keys are missing
            omega = trial_kwargs.get("observability", 0.1)
            conn = trial_kwargs.get("connectivity", 5)
            steps = trial_kwargs.get("steps", 100)
            
            plt.figure(figsize=(10, 5))
            x = np.arange(mean.size)
            plt.fill_between(x, lo, hi, color="green", alpha=0.2, label="10-90% Band")
            plt.plot(x, mean, color="green", linewidth=2, label="Mean Alignment")
            plt.title(f"TVA Aggregate ({trials} trials): Ω={omega}, N={conn}, Steps={steps}")
            plt.ylim(-0.05, 1.05)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()
        except ImportError:
            print("Warning: matplotlib not installed, skipping plot.")
    
    return mean, lo, hi, runs, uncertainty_width

def run_phase_sweep(n_vals, omega_vals, trials=20, steps=100, verbose=True, **kwargs):
    heatmap = np.zeros((len(omega_vals), len(n_vals)))
    uncertainty_map = np.zeros((len(omega_vals), len(n_vals)))
    
    # Protect against collision: remove steps from kwargs if it exists
    # because we pass steps explicitly below.
    kwargs = dict(kwargs)
    kwargs.pop("steps", None)
    
    if verbose:
        print(f"Starting Phase Sweep: {len(n_vals)}x{len(omega_vals)} grid ({trials} trials/point)...")

    for i, omega in enumerate(omega_vals):
        for j, n in enumerate(n_vals):
            mean, _, _, _, width = run_trials(
                trials=trials, connectivity=int(n), observability=omega, steps=steps, 
                plot=False, verbose=False, **kwargs
            )
            heatmap[i, j] = mean[int(steps*0.8):].mean()
            uncertainty_map[i, j] = width[int(steps*0.8):].mean()
        
        if verbose:
            print(f"  > Row {i+1}/{len(omega_vals)} complete (Omega={omega:.2f})")
            
    return heatmap, uncertainty_map