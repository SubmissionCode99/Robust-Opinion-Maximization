# ========================================================= #
# RQ2: SCALABILITY BENCHMARK (CUSTOM VS GUROBI & STEP BREAKDOWN)
# ========================================================= #

import os
import sys
import time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import warnings

# Mute warnings for clean output
warnings.filterwarnings('ignore')

# Ensure Python can find the src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ROMSA.core import (
    get_polyhedral_target,               
    get_polyhedral_target_lp,            
    get_ellipsoidal_uncorrelated_target, 
    get_ellipsoidal_uncorrelated_target_socp, 
    get_ellipsoidal_correlated_target,   
    build_graph_based_echo_chambers,
    get_optimal_stubbornness,            
    evaluate_influence
)

def run_rq2_scalability():
    print("="*85)
    print(" Running RQ2: Scalability Benchmarks")
    print("="*85)

    datasets = ["epinions", "gowalla", "google", "pokec"]
    
    # Store Part 1 Results (Unconstrained: Custom vs Gurobi)
    results_p1 = {
        'poly_custom': [], 'poly_gurobi': [],
        'uncorr_custom': [], 'uncorr_gurobi': []
    }
    
    # Store Part 2 Results (Budget b=1.0: Step 1 vs Step 2)
    results_p2 = {
        'poly_s1': [], 'poly_s2': [], 'poly_tot': [],
        'corr_s1': [], 'corr_s2': [], 'corr_tot': []
    }
    
    dataset_labels = []

    for ds in datasets:
        path = f"data/processed/W_{ds}.npz"
        if not os.path.exists(path):
            print(f"[!] Warning: Could not find graph at {path}. Skipping...")
            continue
            
        W = sp.load_npz(path)
        n = W.shape[0]
        dataset_labels.append(ds.capitalize())
        print(f"\n[*] Processing {ds.capitalize()} (n = {n:,})...")

        # 1. Setup Parameters (Fixed seed for consistency, matching \spara)
        np.random.seed(42)
        s0 = np.random.uniform(0.0, 1.0, n)
        a_init = np.random.uniform(0.0, 1.0, n) # As per updated \spara
        rho = np.random.uniform(0.3, 0.7, n)
        sigma = np.random.uniform(0.3, 0.7, n)
        
        Gamma = 100.0
        Omega = 1.0
        
        # Calculate Initial Influence for Step 2
        print("    -> Computing q_init via LSMR...")
        q_init = evaluate_influence(W, a_init)

        # ========================================================= #
        # PART 1: THE ALGORITHMIC FLEX (UNCONSTRAINED / NO BUDGET)
        # ========================================================= #
        print("    [Part 1] Unconstrained Benchmarks (Custom vs Gurobi)...")
        
        # Polyhedral Unconstrained
        t0 = time.time()
        _ = get_polyhedral_target(s0, Gamma, rho)
        t_poly_custom = time.time() - t0
        
        t0 = time.time()
        _ = get_polyhedral_target_lp(s0, Gamma, rho, q_init=None, b=None)
        t_poly_gurobi = time.time() - t0
        
        results_p1['poly_custom'].append(t_poly_custom)
        results_p1['poly_gurobi'].append(t_poly_gurobi)

        # Ellipsoidal Uncorrelated Unconstrained
        t0 = time.time()
        _ = get_ellipsoidal_uncorrelated_target(s0, Omega, sigma)
        t_uncorr_custom = time.time() - t0
        
        t0 = time.time()
        _ = get_ellipsoidal_uncorrelated_target_socp(s0, Omega, sigma, q_init=None, b=None)
        t_uncorr_gurobi = time.time() - t0
        
        results_p1['uncorr_custom'].append(t_uncorr_custom)
        results_p1['uncorr_gurobi'].append(t_uncorr_gurobi)

        # ========================================================= #
        # PART 2: THE PRACTICAL FLEX (BUDGET b=1.0)
        # ========================================================= #
        print("    [Part 2] Budgeted Full Pipeline (b=1.0)...")
        b_budget = 1.0
        
        # Polyhedral Budgeted Pipeline
        t0 = time.time()
        q_poly_b1 = get_polyhedral_target_lp(s0, Gamma, rho, q_init=q_init, b=b_budget)
        t_poly_s1 = time.time() - t0
        
        t0 = time.time()
        _ = get_optimal_stubbornness(W, q_poly_b1, a_init=a_init)
        t_poly_s2 = time.time() - t0
        
        results_p2['poly_s1'].append(t_poly_s1)
        results_p2['poly_s2'].append(t_poly_s2)
        results_p2['poly_tot'].append(t_poly_s1 + t_poly_s2)

        # Ellipsoidal Correlated Budgeted Pipeline
        t0 = time.time()
        Sigma, Sigma_half = build_graph_based_echo_chambers(W, sigma, target_size=15, corr=0.4)
        q_corr_b1 = get_ellipsoidal_correlated_target(s0, Omega, Sigma_half, q_init=q_init, b=b_budget)
        t_corr_s1 = time.time() - t0
        
        t0 = time.time()
        _ = get_optimal_stubbornness(W, q_corr_b1, a_init=a_init)
        t_corr_s2 = time.time() - t0
        
        results_p2['corr_s1'].append(t_corr_s1)
        results_p2['corr_s2'].append(t_corr_s2)
        results_p2['corr_tot'].append(t_corr_s1 + t_corr_s2)

    if not dataset_labels:
        return

    # ========================================== #
    # GENERATE FIGURE 2: THE ALGORITHMIC FLEX
    # ========================================== #
    print("\n[*] Generating Figure 2 (Log-Scale Bar Charts)...")
    os.makedirs("results/figures", exist_ok=True)
    plt.rcParams.update({'font.size': 14, 'axes.linewidth': 1.5})
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    x = np.arange(len(dataset_labels))
    width = 0.35

    # Panel 1: Polyhedral
    axes[0].bar(x - width/2, results_p1['poly_custom'], width, label='Custom O(n log n)', color='tab:blue', edgecolor='black')
    axes[0].bar(x + width/2, results_p1['poly_gurobi'], width, label='Gurobi (LP)', color='tab:orange', edgecolor='black')
    axes[0].set_yscale('log')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dataset_labels)
    axes[0].set_ylabel("Time (Seconds)")
    axes[0].set_title("Step 1: Polyhedral", fontsize=15, pad=10)
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)
    axes[0].legend(loc='upper left')

    # Panel 2: Ellipsoidal Uncorrelated
    axes[1].bar(x - width/2, results_p1['uncorr_custom'], width, label='Custom O(n log n)', color='tab:blue', edgecolor='black')
    axes[1].bar(x + width/2, results_p1['uncorr_gurobi'], width, label='Gurobi (SOCP)', color='tab:orange', edgecolor='black')
    axes[1].set_yscale('log')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dataset_labels)
    axes[1].set_title("Step 1: Ellipsoidal Uncorrelated", fontsize=15, pad=10)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)
    axes[1].legend(loc='upper left')

    plt.tight_layout()
    plot_path = "results/figures/rq2_algorithmic_speedup.pdf"
    plt.savefig(plot_path, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"    -> Saved to {plot_path}")

    # ========================================== #
    # GENERATE TABLE 2: THE PRACTICAL FLEX
    # ========================================== #
    print("\n" + "="*85)
    print(" GENERATED LATEX TABLE")
    print("="*85)
    
    latex_table = r"""
\begin{table*}[ht]
    \centering
    \caption{Detailed computational runtime (in seconds) of the ROMSA framework across network datasets under a strict intervention budget ($b=1.0$). Time is broken down into computing the influence target (Step 1) and executing the stubbornness $\eta$-search (Step 2).}
    \label{tab:total_runtime}
    \resizebox{0.9\textwidth}{!}{
    \begin{tabular}{l | r r r | r r r}
        \toprule
        \multirow{2}{*}{\textbf{Dataset}} & \multicolumn{3}{c|}{\textbf{Polyhedral}} & \multicolumn{3}{c}{\textbf{Ellipsoidal (Correlated)}} \\
        & Step 1 & Step 2 & \textbf{Total} & Step 1 & Step 2 & \textbf{Total} \\
        \midrule"""
    
    for i, ds in enumerate(dataset_labels):
        # Polyhedral Budgeted
        p_s1 = results_p2['poly_s1'][i]
        p_s2 = results_p2['poly_s2'][i]
        p_tot = results_p2['poly_tot'][i]
        
        # Correlated Budgeted
        c_s1 = results_p2['corr_s1'][i]
        c_s2 = results_p2['corr_s2'][i]
        c_tot = results_p2['corr_tot'][i]
        
        row = f"\n        {ds} & {p_s1:.2f}s & {p_s2:.2f}s & \\textbf{{{p_tot:.2f}s}} & {c_s1:.2f}s & {c_s2:.2f}s & \\textbf{{{c_tot:.2f}s}} \\\\"
        latex_table += row

    latex_table += r"""
        \bottomrule
    \end{tabular}
    }
\end{table*}
"""
    print(latex_table)
    print("="*85)

if __name__ == "__main__":
    run_rq2_scalability()