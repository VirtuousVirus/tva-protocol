import numpy as np
import matplotlib.pyplot as plt

# TVA Engine v1.5
# Features: Input Validation, Vectorized Logic, Monte Carlo Data Return.

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
    if n_agents < 2:
        raise ValueError("n_agents must be >= 2")
    if connectivity < 1:
        raise ValueError("connectivity must be >= 1")
    if not (0.0 <= observability <= 1.0):
        raise ValueError("observability must be in [0.0, 1.0]")
    if not (0.0 <= plasticity <= 1.0):
        raise ValueError("plasticity must be in [0.0, 1.0]")
    if avoid_self_compare and n_agents < 2:
        raise ValueError("avoid_self_compare requires n_agents >= 2")

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
                # The divergent agent is "pruned" (removed) and replaced by 
                # a new, compliant agent with baseline wealth.
                agents[caught] = ALIGNED
                wealth[caught] = replacement_wealth
            else:
                # MODELING CHOICE: Ruin
                # The agent remains but is bankrupted.
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

def run_trials(trials=30, base_seed=42, **kwargs):
    """
    Runs multiple simulations and returns stats.
    Returns: (mean, lo, hi, raw_runs)
    """
    print(f"Running {trials} Monte Carlo trials...")
    
    # Safe kwarg handling (prevent mutation)
    trial_kwargs = dict(kwargs)
    trial_kwargs["plot"] = False
    
    # Run loop with distinct seeds
    runs = np.vstack([
        run_tva_simulation(seed=base_seed + i, **trial_kwargs) 
        for i in range(trials)
    ])
    
    # Stats
    mean = runs.mean(axis=0)
    lo = np.quantile(runs, 0.1, axis=0) # 10th percentile
    hi = np.quantile(runs, 0.9, axis=0) # 90th percentile

    # Plotting
    plt.figure(figsize=(10, 6))
    x = np.arange(mean.size)
    
    # The "Band" (Uncertainty)
    plt.fill_between(x, lo, hi, color="green", alpha=0.2, label="10–90% band")
    # The "Signal" (Mean)
    plt.plot(x, mean, color="green", linewidth=2, label="Mean Alignment")
    
    plt.title(f"TVA Aggregate: Alignment Adoption (Ω={trial_kwargs.get('observability', 0.1)}, N={trial_kwargs.get('connectivity', 5)})")
    plt.xlabel("Time Steps")
    plt.ylabel("% Aligned Agents")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.show()
    
    return mean, lo, hi, runs

if __name__ == "__main__":
    # Default behavior: Run the robust Monte Carlo trial
    run_trials(trials=50, connectivity=5, observability=0.1)
