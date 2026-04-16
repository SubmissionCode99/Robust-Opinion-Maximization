# =========================================================== #
# ====== RQ1 PLOT: WORST-CASE OPINION VS. NOISE BUDGET ====== #
# With Fixed Intervention Budget 'b=1' and 4 Comparison Lines #
# =========================================================== #

import os
import sys
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import warnings

# Mute warnings for clean output
warnings.filterwarnings('ignore')

# Ensure Python can find the src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ROMSA.core import (
    get_polyhedral_target_lp,
    get_ellipsoidal_correlated_target,
    get_worst_case_polyhedral_opinions,
    get_worst_case_ellipsoidal_opinions,
    build_graph_based_echo_chambers,
    evaluate_influence
)

def run_rq1_budgeted_experiment():
    print("="*85)
    print(" Running RQ1: Robustness vs. Adversarial Budget (Fixed Intervention Budget b)")
    print("="*85)

    # Make sure to point this to your actual graph (Epinions for testing, Pokec for final)
    path = "data/processed/W_epinions.npz" 
    if not os.path.exists(path):
        print(f"Error: Could not find graph at {path}")
        return
        
    W = sp.load_npz(path)
    n = W.shape[0]
    print(f"[*] Loaded graph with n={n:,} nodes.")

    # 1. Setup Parameters
    np.random.seed(42) 
    s0 = np.random.uniform(0.0, 1.0, n)
    rho = np.random.uniform(0.3, 0.7, n)
    sigma = np.random.uniform(0.3, 0.7, n)
    
    # Set the fixed intervention budget
    b_budget = 1.0  
    
    # Identify the highest nominal opinion node to find rho_star
    i_star = np.argmax(s0)
    rho_star = rho[i_star]
    
    # 2. Generate Initial State
    print("[*] Generating Initial State...")
    a_init = np.random.uniform(0.1, 0.9, n)
    q_init = evaluate_influence(W, a_init)
    
    # 3. Compute the "Nominal Strategy" under budget b
    print(f"[*] Computing Nominal Strategies (ignoring noise, restricted by b={b_budget})...")
    q_nom_poly = get_polyhedral_target_lp(s0, 0.0, rho, q_init=q_init, b=b_budget)
    
    print("[*] Building Echo Chambers for Ellipsoidal Noise...")
    Sigma, Sigma_half = build_graph_based_echo_chambers(W, sigma, target_size=15, corr=0.4)  # Increase corr from 0.4 higher for more slope in Green and Blue lines
    q_nom_ellip = get_ellipsoidal_correlated_target(s0, 0.0, Sigma_half, q_init=q_init, b=b_budget)

    # 4. Setup Adversarial Axes
    num_points = 8  # Increase to 10-12 for the final paper plot
    
    # Gamma range: 0, EXACTLY rho_star, then log-spaced up to sum(rho)/2
    gamma_vals = np.concatenate(([0, rho_star], np.geomspace(1.0, np.sum(rho)/2, num_points - 2)))
    omega_vals = np.linspace(0, 1.0 / np.max(sigma), num_points)

    # Data structures for the 4 lines
    results_poly = {'Init': [], 'Nominal': [], 'Robust_WC': [], 'Robust_ZN': []}
    results_ellip = {'Init': [], 'Nominal': [], 'Robust_WC': [], 'Robust_ZN': []}

    print("\n--- Evaluating Polyhedral Noise ---")
    for i, gamma in enumerate(gamma_vals):
        print(f"  -> Solving Polyhedral gamma {i+1}/{num_points} ({gamma:,.2f})...")
        
        q_romsa = get_polyhedral_target_lp(s0, gamma, rho, q_init=q_init, b=b_budget)
        
        wc_init = q_init @ get_worst_case_polyhedral_opinions(q_init, s0, gamma, rho)
        wc_nom = q_nom_poly @ get_worst_case_polyhedral_opinions(q_nom_poly, s0, gamma, rho)
        wc_romsa = q_romsa @ get_worst_case_polyhedral_opinions(q_romsa, s0, gamma, rho)
        zn_romsa = q_romsa @ s0
        
        results_poly['Init'].append(wc_init)
        results_poly['Nominal'].append(wc_nom)
        results_poly['Robust_WC'].append(wc_romsa)
        results_poly['Robust_ZN'].append(zn_romsa)

    print("\n--- Evaluating Ellipsoidal Correlated Noise ---")
    for i, omega in enumerate(omega_vals):
        print(f"  -> Solving Ellipsoidal omega {i+1}/{num_points} ({omega:.3f})...")
        
        q_romsa = get_ellipsoidal_correlated_target(s0, omega, Sigma_half, q_init=q_init, b=b_budget)
        
        wc_init = q_init @ get_worst_case_ellipsoidal_opinions(q_init, s0, omega, Sigma)
        wc_nom = q_nom_ellip @ get_worst_case_ellipsoidal_opinions(q_nom_ellip, s0, omega, Sigma)
        wc_romsa = q_romsa @ get_worst_case_ellipsoidal_opinions(q_romsa, s0, omega, Sigma)
        zn_romsa = q_romsa @ s0
        
        results_ellip['Init'].append(wc_init)
        results_ellip['Nominal'].append(wc_nom)
        results_ellip['Robust_WC'].append(wc_romsa)
        results_ellip['Robust_ZN'].append(zn_romsa)

    # ========================================== #
    # 5. Generate the Plot
    # ========================================== #
    print("\n--- Generating PDF Plot ---")
    os.makedirs("results/figures", exist_ok=True)
    plt.rcParams.update({'font.size': 14, 'axes.linewidth': 1.5})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # ---------------- Subplot 1: Polyhedral ----------------
    axes[0].plot(gamma_vals, results_poly['Init'], marker='s', linestyle=':', color='tab:gray', linewidth=2, label='Initial Strategy')
    axes[0].plot(gamma_vals, results_poly['Nominal'], marker='o', linestyle='--', color='tab:red', linewidth=2, label='Nominal Strategy')
    axes[0].plot(gamma_vals, results_poly['Robust_WC'], marker='D', linestyle='-', color='tab:blue', linewidth=2.5, label='Robust (Worst-Case)')
    axes[0].plot(gamma_vals, results_poly['Robust_ZN'], marker='^', linestyle='-.', color='tab:green', linewidth=2, label='Robust (Zero Noise)')
    
    # Format Polyhedral X-Axis
    axes[0].axvline(x=rho_star, color='black', linestyle=':', alpha=0.4)
    axes[0].set_xscale('symlog', linthresh=1.0)
    axes[0].set_xlim(left=-0.1, right=np.max(gamma_vals) * 1.5)
    
    # Custom Ticks mapping
    max_pow = int(np.floor(np.log10(np.max(gamma_vals))))
    custom_ticks = [0, rho_star] + [10**i for i in range(0, max_pow + 1)]
    custom_labels = ['0', r'$\rho_{i^*}$'] + [r'$10^{%d}$' % i for i in range(0, max_pow + 1)]
    axes[0].set_xticks(custom_ticks)
    axes[0].set_xticklabels(custom_labels)

    axes[0].set_xlabel(r"Budget ($\gamma$)")
    axes[0].set_ylabel(r"Public Opinion")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_title(f"Polyhedral Noise ($b={b_budget}$)", fontsize=15, pad=10)

    # ---------------- Subplot 2: Ellipsoidal ----------------
    axes[1].plot(omega_vals, results_ellip['Init'], marker='s', linestyle=':', color='tab:gray', linewidth=2, label='Initial Strategy')
    axes[1].plot(omega_vals, results_ellip['Nominal'], marker='o', linestyle='--', color='tab:red', linewidth=2, label='Nominal Strategy')
    axes[1].plot(omega_vals, results_ellip['Robust_WC'], marker='D', linestyle='-', color='tab:blue', linewidth=2.5, label='Robust (Worst-Case)')
    axes[1].plot(omega_vals, results_ellip['Robust_ZN'], marker='^', linestyle='-.', color='tab:green', linewidth=2, label='Robust (Zero Noise)')
    
    axes[1].set_xlabel(r"Budget ($\omega$)")
    axes[1].set_ylabel(r"Public Opinion")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_title(f"Ellipsoidal Noise ($b={b_budget}$)", fontsize=15, pad=10)
    
    # Custom Legend Reordering to match visual hierarchy
    handles, labels = axes[1].get_legend_handles_labels()
    order = [
        labels.index('Robust (Zero Noise)'),
        labels.index('Robust (Worst-Case)'), 
        labels.index('Nominal Strategy'),
        labels.index('Initial Strategy')
    ]
    axes[1].legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower left', fontsize=12)

    plt.tight_layout()
    plot_path = "results/figures/rq1_robustness_budgeted.pdf"
    plt.savefig(plot_path, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Success! Plot saved to {plot_path}")

if __name__ == "__main__":
    run_rq1_budgeted_experiment()