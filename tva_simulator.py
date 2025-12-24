import numpy as np
import matplotlib.pyplot as plt

# TVA Core Logic V1: Veracity vs. Entropy
# Demonstrates the "Maintenance Wall" where high-entropy strategies (deception) fail.

def run_tva_simulation(n_agents=100, steps=50, connectivity=5, observability=0.1):
    # 0 = Aligned (Veracity), 1 = Divergent (Deceptive)
    agents = np.random.choice([0, 1], size=n_agents, p=[0.5, 0.5])
    wealth = np.ones(n_agents) * 10.0
    
    history_alignment = []
    
    for t in range(steps):
        # The Maintenance Wall: Entropy cost scales with connectivity^2
        entropy_cost = (connectivity ** 2) * 0.05
        
        # The Pruning Threshold: Probability of failure scales with observability
        pruning_prob = 1 - (1 - observability) ** connectivity
        
        new_strategies = agents.copy()
        
        for i in range(n_agents):
            if agents[i] == 1: # Divergent (Deceptive)
                # Pay the entropy tax
                wealth[i] -= entropy_cost
                
                # Check for Pruning (Scandal/Ruin)
                if np.random.random() < pruning_prob:
                    wealth[i] = 0 # Pruned
                    new_strategies[i] = 0 # Forced Alignment
                else:
                    # Deception Alpha (Short term gain)
                    wealth[i] += 2.0 
            else:
                # Aligned Strategy (Veracity Dividend)
                wealth[i] += 1.0 # Steady growth, zero entropy tax
        
        # Stochastic Replicator: Copy the wealthy
        avg_wealth_aligned = np.mean(wealth[agents == 0]) if np.any(agents == 0) else 0
        avg_wealth_divergent = np.mean(wealth[agents == 1]) if np.any(agents == 1) else 0
        
        for i in range(n_agents):
            if wealth[i] < (avg_wealth_aligned if agents[i] == 1 else avg_wealth_divergent):
                # Switch strategy if underperforming
                if np.random.random() < 0.1: # Plasticity
                    new_strategies[i] = 1 - new_strategies[i]
        
        agents = new_strategies
        history_alignment.append(np.mean(agents == 0))

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(history_alignment, label='Veracity Alignment')
    plt.title(f'TVA Phase 1: Alignment Adoption (Obs={observability}, Conn={connectivity})')
    plt.xlabel('Time Steps')
    plt.ylabel('% Aligned Agents')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_tva_simulation()
