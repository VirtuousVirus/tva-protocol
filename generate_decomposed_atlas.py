import numpy as np
import matplotlib.pyplot as plt
from tva_simulator import run_phase_sweep

def plot_decomposed_atlas(maps, n_vals, omega_vals):
    # maps: {'std', 'inc', 'soc'}, each containing 'mean' and 'unc'
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    
    def plot_sub(ax, data, title, cmap, vmin, vmax, label):
        im = ax.imshow(data, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label=label)

    # --- ROW 1: Mean Outcomes ---
    plot_sub(axes[0,0], maps['std']['mean'], 
             "A. Baseline (Std Incentives + Std Pressure)", 
             'RdYlGn', 0, 1, "Mean Alignment")
    plot_sub(axes[0,1], maps['inc']['mean'], 
             "B. Incentive Swap (Structure vs Pressure)", 
             'RdYlGn', 0, 1, "Mean Alignment")
    plot_sub(axes[0,2], maps['soc']['mean'], 
             "C. Social Inversion (Structure vs Pressure)", 
             'RdYlGn', 0, 1, "Mean Alignment")

    # --- ROW 2: Fragility Gaps ---
    plot_sub(axes[1,0], maps['std']['unc'], 
             "D. Baseline Uncertainty (Ref)", 
             'plasma', 0, 0.4, "Sigma")

    # Gap E: Incentive Fragility
    gap_inc = maps['inc']['unc'] - maps['std']['unc']
    plot_sub(axes[1,1], gap_inc, 
             "E. Incentive Swap Fragility (Role-Swap Stress)", 
             'magma', -0.1, 0.4, "Excess Entropy")

    # Gap F: Social Fragility
    gap_soc = maps['soc']['unc'] - maps['std']['unc']
    plot_sub(axes[1,2], gap_soc, 
             "F. Social Fragility (Cost of Fighting Pressure)", 
             'magma', -0.1, 0.4, "Excess Entropy")

    plt.suptitle("TVA Decomposed Atlas: Orthogonal Stress Tests", fontsize=16)
    for ax in axes[:,0]: ax.set_ylabel(r"Observability ($\Omega$)")
    for ax in axes[-1,:]: ax.set_xlabel(r"Connectivity ($N$)")
    
    plt.tight_layout()
    plt.savefig("tva_atlas_decomposed.png")
    print("Saved Decomposed Atlas to tva_atlas_decomposed.png")

if __name__ == "__main__":
    n_range = np.arange(2, 42, 2)
    omega_range = np.linspace(0, 0.5, 20)
    kw = dict(trials=30, n_agents=100, steps=150, break_even_connectivity=20, verbose=True)
    
    maps = {'std': {}, 'inc': {}, 'soc': {}}
    
    print("1. Running Baseline...")
    maps['std']['mean'], maps['std']['unc'] = run_phase_sweep(
        n_range, omega_range, invert_payoffs=False, invert_pruning=False, **kw
    )
    
    print("2. Running Incentive Swap...")
    maps['inc']['mean'], maps['inc']['unc'] = run_phase_sweep(
        n_range, omega_range, invert_payoffs=True, invert_pruning=False, **kw
    )
    
    print("3. Running Social Pressure Inversion...")
    maps['soc']['mean'], maps['soc']['unc'] = run_phase_sweep(
        n_range, omega_range, invert_payoffs=False, invert_pruning=True, **kw
    )
    
    plot_decomposed_atlas(maps, n_range, omega_range)