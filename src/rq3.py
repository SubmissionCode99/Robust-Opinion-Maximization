# ========================================================= #
# RQ3 PLOT: STUBBORNNESS DISTRIBUTION (UNBUDGETED VS BUDGETED)
# ========================================================= #

# import os
# import sys
# import numpy as np
# import scipy.sparse as sp
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
# import warnings

# # Mute warnings for clean output
# warnings.filterwarnings('ignore')

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# from src.ROMSA.core import (
#     get_polyhedral_target_lp,
#     get_ellipsoidal_correlated_target,
#     build_graph_based_echo_chambers,
#     evaluate_influence,
#     get_optimal_stubbornness
# )

# def run_rq3_combined_stubbornness_distribution():
#     print("="*85)
#     print(" Running RQ3: Combined Stubbornness Analysis (4x3 Grid)")
#     print("="*85)

#     path = "data/processed/W_epinions.npz" 
#     if not os.path.exists(path):
#         print(f"Error: Could not find graph at {path}")
#         return
        
#     W = sp.load_npz(path)
#     n = W.shape[0]
#     print(f"[*] Loaded graph with n={n:,} nodes.")

#     # 1. Setup Parameters 
#     np.random.seed(42) 
#     s0 = np.random.uniform(0.0, 1.0, n)
#     a_init = np.random.uniform(0.0, 1.0, n)
#     rho = np.random.uniform(0.3, 0.7, n)
#     sigma = np.random.uniform(0.3, 0.7, n)
    
#     b_budget = 1.0
#     Gamma = 100.0  
#     Omega = 1.0

#     # Calculate Degree 
#     adjacency = W.astype(bool).astype(int)
#     in_degree = np.array(adjacency.sum(axis=0)).flatten()
#     out_degree = np.array(adjacency.sum(axis=1)).flatten()
#     degree = in_degree + out_degree
    
#     # 2. Get Initial State
#     print("[*] Computing initial influence q_init...")
#     q_init = evaluate_influence(W, a_init)

#     print("[*] Building Echo Chambers via METIS...")
#     Sigma, Sigma_half = build_graph_based_echo_chambers(W, sigma, target_size=15, corr=0.4)

#     # ========================================== #
#     # 3. SOLVE UNBUDGETED SCENARIOS
#     # ========================================== #
#     print("\n--- Solving Unbudgeted Scenarios (b=\u221E) ---")
#     q_poly_un = get_polyhedral_target_lp(s0, Gamma, rho, q_init=None, b=None)
#     q_ellip_un = get_ellipsoidal_correlated_target(s0, Omega, Sigma_half, q_init=None, b=None)

#     a_poly_un = get_optimal_stubbornness(W, q_poly_un, a_init=None)
#     a_ellip_un = get_optimal_stubbornness(W, q_ellip_un, a_init=None)

#     # ========================================== #
#     # 4. SOLVE BUDGETED SCENARIOS
#     # ========================================== #
#     print("\n--- Solving Budgeted Scenarios (b=1.0) ---")
#     q_poly_b1 = get_polyhedral_target_lp(s0, Gamma, rho, q_init=q_init, b=b_budget)
#     q_ellip_b1 = get_ellipsoidal_correlated_target(s0, Omega, Sigma_half, q_init=q_init, b=b_budget)

#     a_poly_b1 = get_optimal_stubbornness(W, q_poly_b1, a_init=a_init)
#     a_ellip_b1 = get_optimal_stubbornness(W, q_ellip_b1, a_init=a_init)

#     # ========================================== #
#     # 5. Generate the 4x3 Plot
#     # ========================================== #
#     print("\n[*] Generating 4x3 PDF Plot...")
#     os.makedirs("results/figures", exist_ok=True)
#     plt.rcParams.update({'font.size': 13, 'axes.linewidth': 1.5})
    
#     # Make the figure taller to comfortably fit 4 rows
#     fig, axes = plt.subplots(4, 3, figsize=(16, 20), dpi=200)
    
#     def plot_scatter(ax, x_data, y_data, x_label, title, color):
#         # Filter active nodes (removes the "noise" of nodes that received 0 influence)
#         # Also strictly filters out x <= 0 to prevent log domain errors
#         active_mask = (y_data > 1e-6) & (x_data > 0)
#         x_active = x_data[active_mask]
#         y_active = y_data[active_mask]
        
#         if len(x_active) > 1:
#             # Transform Y into log space (because set_yscale is 'log')
#             y_fit = np.log10(y_active)
            
#             # Transform X into log space ONLY if it's the Degree plot
#             if "Degree" in x_label:
#                 x_fit = np.log10(x_active)
#             else:
#                 x_fit = x_active
                
#             corr, _ = pearsonr(x_fit, y_fit)
#             corr_text = f"Pearson = {corr:.2f}"
#         else:
#             corr_text = "N/A"
            
#         ax.scatter(x_active, y_active, alpha=0.3, s=15, c=color, edgecolors='none', label=corr_text)
#         ax.set_yscale('log')
        
#         if "Degree" in x_label:
#             ax.set_xscale('log')
            
#         ax.set_xlabel(x_label)
#         ax.set_ylabel(r"Optimal Stubbornness ($a_i^*$)")
#         ax.set_title(title, fontsize=14, pad=10)
#         ax.grid(True, linestyle='--', alpha=0.5)
#         ax.legend(loc='upper right', handletextpad=0.0, handlelength=0)

#     # --- ROW 1: Polyhedral Unbudgeted ---
#     plot_scatter(axes[0, 0], s0, a_poly_un, r"Nominal Opinion ($s_i$)", "Polyhedral (Unbudgeted): a* vs. Opinion", 'tab:blue')
#     plot_scatter(axes[0, 1], degree, a_poly_un, "Node Degree", "Polyhedral (Unbudgeted): a* vs. Degree", 'tab:blue')
#     plot_scatter(axes[0, 2], rho, a_poly_un, r"Local Noise ($\rho_i$)", "Polyhedral (Unbudgeted): a* vs. Noise", 'tab:blue')

#     # --- ROW 2: Ellipsoidal Unbudgeted ---
#     plot_scatter(axes[1, 0], s0, a_ellip_un, r"Nominal Opinion ($s_i$)", "Ellipsoidal (Unbudgeted): a* vs. Opinion", 'tab:orange')
#     plot_scatter(axes[1, 1], degree, a_ellip_un, "Node Degree", "Ellipsoidal (Unbudgeted): a* vs. Degree", 'tab:orange')
#     plot_scatter(axes[1, 2], sigma, a_ellip_un, r"Local Noise ($\sigma_i$)", "Ellipsoidal (Unbudgeted): a* vs. Variance", 'tab:orange')

#     # --- ROW 3: Polyhedral Budgeted (b=1.0) ---
#     plot_scatter(axes[2, 0], s0, a_poly_b1, r"Nominal Opinion ($s_i$)", "Polyhedral ($b=1.0$): a* vs. Opinion", 'tab:blue')
#     plot_scatter(axes[2, 1], degree, a_poly_b1, "Node Degree", "Polyhedral ($b=1.0$): a* vs. Degree", 'tab:blue')
#     plot_scatter(axes[2, 2], rho, a_poly_b1, r"Local Noise ($\rho_i$)", "Polyhedral ($b=1.0$): a* vs. Noise", 'tab:blue')

#     # --- ROW 4: Ellipsoidal Budgeted (b=1.0) ---
#     plot_scatter(axes[3, 0], s0, a_ellip_b1, r"Nominal Opinion ($s_i$)", "Ellipsoidal ($b=1.0$): a* vs. Opinion", 'tab:orange')
#     plot_scatter(axes[3, 1], degree, a_ellip_b1, "Node Degree", "Ellipsoidal ($b=1.0$): a* vs. Degree", 'tab:orange')
#     plot_scatter(axes[3, 2], sigma, a_ellip_b1, r"Local Noise ($\sigma_i$)", "Ellipsoidal ($b=1.0$): a* vs. Variance", 'tab:orange')

#     # Adjust vertical spacing so the 4 rows don't crash into each other
#     plt.tight_layout()
#     fig.subplots_adjust(hspace=0.4)
    
#     plot_path = "results/figures/rq3_stubbornness_distribution.pdf"
#     plt.savefig(plot_path, bbox_inches='tight', format='pdf')
#     plt.close()
#     print(f"[*] Success! Plot saved to {plot_path}")

# if __name__ == "__main__":
#     run_rq3_combined_stubbornness_distribution()
    
# ========================================================= #
# RQ3 PLOT: THE ECHO CHAMBER EFFECT (BINNED BAR CHART)
# (Visualizing Divestment across the Toxicity Spectrum)
# ========================================================= #

# import os
# import sys
# import numpy as np
# import scipy.sparse as sp
# import matplotlib.pyplot as plt
# import warnings

# # Mute warnings for clean output
# warnings.filterwarnings('ignore')

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# from src.ROMSA.core import (
#     get_ellipsoidal_uncorrelated_target_socp,
#     get_ellipsoidal_correlated_target,
#     build_graph_based_echo_chambers,
#     evaluate_influence
# )

# def run_rq3_binned_echo_chamber_effect():
#     print("="*85)
#     print(" Running RQ3: Binned Echo Chamber Divestment (Pure METIS)")
#     print("="*85)

#     path = "data/processed/W_epinions.npz" 
#     if not os.path.exists(path):
#         return
        
#     W = sp.load_npz(path)
#     n = W.shape[0]

#     np.random.seed(42) 
#     s0 = np.random.uniform(0.0, 1.0, n)
#     a_init = np.random.uniform(0.0, 1.0, n)
#     sigma = np.random.uniform(0.3, 0.7, n)
    
#     b_budget = 1.0
#     Omega = 1.0
    
#     print("[*] Computing initial influence q_init...")
#     q_init = evaluate_influence(W, a_init)
    
#     print(f"[*] Solving Budgeted Uncorrelated Ellipsoidal SOCP (b={b_budget})...")
#     q_uncorr = get_ellipsoidal_uncorrelated_target_socp(s0, Omega, sigma, q_init=q_init, b=b_budget)

#     print("[*] Building Pure METIS Echo Chambers...")
#     Sigma, Sigma_half = build_graph_based_echo_chambers(W, sigma, target_size=15, corr=0.4)
    
#     print(f"[*] Solving Budgeted Correlated Ellipsoidal SOCP (b={b_budget})...")
#     q_corr = get_ellipsoidal_correlated_target(s0, Omega, Sigma_half, q_init=q_init, b=b_budget)

#     # 1. Calculate Metrics
#     print("[*] Calculating Binned Metrics...")
#     sigma_diag = Sigma.diagonal()
#     sigma_row_sums = np.array(Sigma.sum(axis=1)).flatten()
#     correlation_strength = sigma_row_sums - sigma_diag
    
#     delta_q = q_corr - q_uncorr

#     # Filter out nodes that effectively received zero influence in both scenarios to avoid noise
#     tolerance = 1e-6
#     active_mask = (q_uncorr > tolerance) | (q_corr > tolerance)
    
#     active_corr = correlation_strength[active_mask]
#     active_delta = delta_q[active_mask]

#     # 2. Bin the Data (10 Equal-width bins based on correlation strength)
#     num_bins = 10
#     bins = np.linspace(active_corr.min(), active_corr.max(), num_bins + 1)
    
#     bin_centers = []
#     bin_means = []
    
#     for i in range(num_bins):
#         # Find nodes that fall into the current bin
#         mask = (active_corr >= bins[i]) & (active_corr <= bins[i+1])
#         if np.any(mask):
#             bin_centers.append((bins[i] + bins[i+1]) / 2)
#             bin_means.append(np.mean(active_delta[mask]))

#     # ========================================== #
#     # 3. Generate the Bar Chart
#     print("\n[*] Generating Plot...")
#     os.makedirs("results/figures", exist_ok=True)
#     plt.rcParams.update({'font.size': 14, 'axes.linewidth': 1.5})
    
#     fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

#     colors = ['tab:blue' if val > 0 else 'tab:red' for val in bin_means]
    
#     # Calculate width of bars based on bin spacing
#     bar_width = (bins[1] - bins[0]) * 0.8
    
#     ax.bar(bin_centers, bin_means, width=bar_width, color=colors, edgecolor='black', alpha=0.8)
#     ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=0)

#     # Use symlog for Y-axis to handle the magnitudes gracefully
#     ax.set_yscale('symlog', linthresh=1e-6)
    
#     ax.set_xlabel(r"Echo Chamber Toxicity (Correlation Strength $\sum_{j \neq i} \Sigma_{ij}$)")
#     ax.set_ylabel(r"Average Shift in Influence ($\overline{\Delta q_i^*}$)")
#     ax.set_title("Systematic Divestment from Highly Correlated Echo Chambers", fontsize=16, pad=15)
#     ax.grid(True, linestyle='--', alpha=0.5)

#     # Custom Legend
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], color='tab:blue', lw=6, label='Net Reinvestment (Gained Influence)'),
#         Line2D([0], [0], color='tab:red', lw=6, label='Net Divestment (Lost Influence)')
#     ]
#     ax.legend(handles=legend_elements, loc='upper right')

#     plt.tight_layout()
#     plot_path = "results/figures/rq3_binned_correlation_effect.pdf"
#     plt.savefig(plot_path, bbox_inches='tight', format='pdf')
#     plt.close()
#     print(f"[*] Success! Plot saved to {plot_path}")

# if __name__ == "__main__":
#     run_rq3_binned_echo_chamber_effect()

# ========================================================= #
# RQ3 PLOT: NUMBER OF ACTIVE NODES VS. NOISE BUDGET (2x2 GRID)
# (Comparing Unconstrained vs. Budgeted Diversification)
# ========================================================= #

# import os
# import sys
# import numpy as np
# import scipy.sparse as sp
# import matplotlib.pyplot as plt
# import warnings

# # Mute warnings for clean output
# warnings.filterwarnings('ignore')

# # Ensure Python can find the src module
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# from src.ROMSA.core import (
#     get_polyhedral_target_lp,
#     get_ellipsoidal_correlated_target,
#     build_graph_based_echo_chambers,
#     evaluate_influence
# )

# def run_rq3_combined_active_nodes_experiment():
#     print("="*85)
#     print(" Running RQ3: 2x2 Diversification Analysis (Unconstrained vs b=1.0)")
#     print("="*85)

#     path = "data/processed/W_epinions.npz" 
#     if not os.path.exists(path):
#         print(f"Error: Could not find graph at {path}")
#         return
        
#     W = sp.load_npz(path)
#     n = W.shape[0]
#     print(f"[*] Loaded graph with n={n:,} nodes.")

#     # 1. Setup Parameters 
#     np.random.seed(42) 
#     s0 = np.random.uniform(0.0, 1.0, n)
#     a_init = np.random.uniform(0.0, 1.0, n)
#     rho = np.random.uniform(0.3, 0.7, n)
#     sigma = np.random.uniform(0.3, 0.7, n)
    
#     b_budget = 1.0
    
#     # Identify the highest nominal opinion node to find rho_star
#     i_star = np.argmax(s0)
#     rho_star = rho[i_star]
    
#     print("[*] Computing initial influence q_init...")
#     q_init = evaluate_influence(W, a_init)
    
#     print("[*] Building Echo Chambers for Ellipsoidal Noise...")
#     Sigma, Sigma_half = build_graph_based_echo_chambers(W, sigma, target_size=15, corr=0.4)

#     # 3. Setup SHARED Adversarial Axes
#     num_points = 15  
#     gamma_max = np.sum(rho)
#     gamma_vals = np.concatenate(([0, rho_star], np.geomspace(1.0, gamma_max, num_points - 2)))
#     omega_vals = np.linspace(0, 1.0 / np.max(sigma), num_points)

#     # Storage arrays
#     active_poly_unc = []
#     active_ellip_unc = []
#     active_poly_con = []
#     active_ellip_con = []
    
#     tolerance = 1e-6 

#     print("\n--- Solving Polyhedral Noise ---")
#     for i, gamma in enumerate(gamma_vals):
#         print(f"  -> gamma {i+1}/{num_points} ({gamma:,.2f})")
#         # Unconstrained
#         q_p_unc = get_polyhedral_target_lp(s0, gamma, rho, q_init=None, b=None)
#         active_poly_unc.append(np.sum(q_p_unc > tolerance))
        
#         # Constrained
#         q_p_con = get_polyhedral_target_lp(s0, gamma, rho, q_init=q_init, b=b_budget)
#         active_poly_con.append(np.sum(q_p_con > tolerance))

#     print("\n--- Solving Ellipsoidal Correlated Noise ---")
#     for i, omega in enumerate(omega_vals):
#         print(f"  -> omega {i+1}/{num_points} ({omega:.3f})")
#         # Unconstrained
#         q_e_unc = get_ellipsoidal_correlated_target(s0, omega, Sigma_half, q_init=None, b=None)
#         active_ellip_unc.append(np.sum(q_e_unc > tolerance))
        
#         # Constrained
#         q_e_con = get_ellipsoidal_correlated_target(s0, omega, Sigma_half, q_init=q_init, b=b_budget)
#         active_ellip_con.append(np.sum(q_e_con > tolerance))

#     # ========================================== #
#     # 4. Generate the 2x2 Plot
#     print("\n[*] Generating 2x2 PDF Plot...")
#     os.makedirs("results/figures", exist_ok=True)
#     plt.rcParams.update({'font.size': 14, 'axes.linewidth': 1.5})
    
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)

#     # Helper function for Polyhedral x-axis formatting
#     def format_poly_xaxis(ax):
#         ax.axvline(x=rho_star, color='red', linestyle=':', alpha=0.5, label=r'$\rho_{i^*}$')
#         ax.set_xscale('symlog', linthresh=1.0)
#         ax.set_xlim(left=-0.1, right=np.max(gamma_vals) * 1.5)
#         max_pow = int(np.floor(np.log10(np.max(gamma_vals))))
#         custom_ticks = [0, rho_star] + [10**i for i in range(0, max_pow + 1)]
#         custom_labels = ['0', r'$\rho_{i^*}$'] + [r'$10^{%d}$' % i for i in range(0, max_pow + 1)]
#         ax.set_xticks(custom_ticks)
#         ax.set_xticklabels(custom_labels)
#         ax.grid(True, linestyle='--', alpha=0.7)

#     # --- TOP LEFT: Polyhedral Unconstrained ---
#     axes[0, 0].plot(gamma_vals, active_poly_unc, marker='o', linestyle='-', color='black', linewidth=2)
#     format_poly_xaxis(axes[0, 0])
#     axes[0, 0].set_ylim(bottom=0)
#     axes[0, 0].set_ylabel(r"Active Nodes ($q_i^* > 0$)")
#     axes[0, 0].set_title("Polyhedral (Unconstrained)", fontsize=15, pad=10)
#     axes[0, 0].legend(loc='upper left')

#     # --- TOP RIGHT: Ellipsoidal Unconstrained ---
#     axes[0, 1].plot(omega_vals, active_ellip_unc, marker='s', linestyle='-', color='black', linewidth=2)
#     axes[0, 1].set_ylim(bottom=0)
#     axes[0, 1].grid(True, linestyle='--', alpha=0.7)
#     axes[0, 1].set_title("Ellipsoidal (Unconstrained)", fontsize=15, pad=10)

#     # --- BOTTOM LEFT: Polyhedral Constrained (b=1.0) ---
#     axes[1, 0].plot(gamma_vals, active_poly_con, marker='o', linestyle='-', color='black', linewidth=2)
#     format_poly_xaxis(axes[1, 0])
#     axes[1, 0].set_xlabel(r"Adversary Budget ($\gamma$)")
#     axes[1, 0].set_ylabel(r"Active Nodes ($q_i^* > 0$)")
#     axes[1, 0].set_title(f"Polyhedral ($b={b_budget}$)", fontsize=15, pad=10)
#     axes[1, 0].legend(loc='upper left')

#     # --- BOTTOM RIGHT: Ellipsoidal Constrained (b=1.0) ---
#     axes[1, 1].plot(omega_vals, active_ellip_con, marker='s', linestyle='-', color='black', linewidth=2)
#     axes[1, 1].set_xlabel(r"Adversary Budget ($\omega$)")
#     axes[1, 1].grid(True, linestyle='--', alpha=0.7)
#     axes[1, 1].set_title(f"Ellipsoidal ($b={b_budget}$)", fontsize=15, pad=10)

#     plt.tight_layout()
#     plot_path = "results/figures/rq3_active_nodes_2x2.pdf"
#     plt.savefig(plot_path, bbox_inches='tight', format='pdf')
#     plt.close()
#     print(f"[*] Success! Plot saved to {plot_path}")

# if __name__ == "__main__":
#     run_rq3_combined_active_nodes_experiment()


# ========================================================= #
# RQ3 PLOT: INFLUENCE DISTRIBUTION VS NODE FEATURES 2 by 3  #
# ========================================================= #

import os
import sys
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings

# Mute warnings for clean output
warnings.filterwarnings('ignore')

# Ensure Python can find the src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ROMSA.core import (
    get_polyhedral_target_lp,
    get_ellipsoidal_correlated_target,
    build_graph_based_echo_chambers,
    evaluate_influence
)

def run_rq3_influence_distribution():
    print("="*85)
    print(" Running RQ3: Influence Distribution Analysis (Epinions)")
    print("="*85)

    path = "data/processed/W_epinions.npz" 
    if not os.path.exists(path):
        print(f"Error: Could not find graph at {path}")
        return
        
    W = sp.load_npz(path)
    n = W.shape[0]
    print(f"[*] Loaded graph with n={n:,} nodes.")

    # 1. Setup Parameters (matching Section 6.1)
    np.random.seed(42) 
    s0 = np.random.uniform(0.0, 1.0, n)
    a_init = np.random.uniform(0.0, 1.0, n)
    rho = np.random.uniform(0.3, 0.7, n)
    sigma = np.random.uniform(0.3, 0.7, n)
    
    b_budget = 2.0
    Gamma = 100.0
    Omega = 1.0

    # Calculate Degree (Total edges connected to the node)
    # W is row stochastic, so we look at the adjacency structure
    adjacency = W.astype(bool).astype(int)
    in_degree = np.array(adjacency.sum(axis=0)).flatten()
    out_degree = np.array(adjacency.sum(axis=1)).flatten()
    degree = in_degree + out_degree
    
    # 2. Get Initial State
    print("[*] Computing initial influence q_init...")
    q_init = evaluate_influence(W, a_init)

    # 3. Solve for Targets
    print(f"[*] Solving Budgeted Polyhedral LP (b={b_budget}, Gamma={Gamma})...")
    q_poly = get_polyhedral_target_lp(s0, Gamma, rho, q_init=q_init, b=b_budget)

    print(f"[*] Solving Budgeted Ellipsoidal SOCP (b={b_budget}, Omega={Omega})...")
    Sigma, Sigma_half = build_graph_based_echo_chambers(W, sigma, target_size=15, corr=0.4)
    q_ellip = get_ellipsoidal_correlated_target(s0, Omega, Sigma_half, q_init=q_init, b=b_budget)

    # ========================================== #
    # 4. Generate the Plot
    # ========================================== #
    print("\n[*] Generating 2x3 PDF Plot...")
    os.makedirs("results/figures", exist_ok=True)
    plt.rcParams.update({'font.size': 13, 'axes.linewidth': 1.5})
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=200)
    
    # Helper function to plot and calculate correlation (only for active nodes to avoid zero-distortion on log scale)
    def plot_scatter(ax, x_data, y_data, x_label, title, color):
        # Filter strictly active nodes and positive x_data to safely use log scale
        active_mask = (y_data > 1e-8) & (x_data > 0)
        x_active = x_data[active_mask]
        y_active = y_data[active_mask]
        
        # Calculate Pearson Correlation on active nodes in log space where applicable
        if len(x_active) > 1:
            y_fit = np.log10(y_active)
            
            # If x-axis is degree, use log scale for X correlation
            if "Degree" in x_label:
                x_fit = np.log10(x_active)
            else:
                x_fit = x_active
                
            corr, _ = pearsonr(x_fit, y_fit)
            corr_text = f"Pearson = {corr:.2f}"
        else:
            corr_text = "N/A"
            
        ax.scatter(x_active, y_active, alpha=0.3, s=15, c=color, edgecolors='none', label=corr_text)
        ax.set_yscale('log')
        
        # If x-axis is degree, use log scale for X visual axes as well
        if "Degree" in x_label:
            ax.set_xscale('log')
            
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"Target Influence ($q_i^*$)")
        ax.set_title(title, fontsize=14, pad=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', handletextpad=0.0, handlelength=0)

    # --- ROW 1: Polyhedral Noise ---
    plot_scatter(axes[0, 0], s0, q_poly, r"Nominal Opinion ($s_i$)", "Polyhedral: Influence vs. Opinion", 'tab:blue')
    plot_scatter(axes[0, 1], degree, q_poly, "Node Degree", "Polyhedral: Influence vs. Degree", 'tab:blue')
    plot_scatter(axes[0, 2], rho, q_poly, r"Local Noise ($\rho_i$)", "Polyhedral: Influence vs. Noise Bounds", 'tab:blue')

    # --- ROW 2: Ellipsoidal Correlated Noise ---
    plot_scatter(axes[1, 0], s0, q_ellip, r"Nominal Opinion ($s_i$)", "Ellipsoidal: Influence vs. Opinion", 'tab:orange')
    plot_scatter(axes[1, 1], degree, q_ellip, "Node Degree", "Ellipsoidal: Influence vs. Degree", 'tab:orange')
    plot_scatter(axes[1, 2], sigma, q_ellip, r"Local Noise ($\sigma_i$)", "Ellipsoidal: Influence vs. Variance", 'tab:orange')

    plt.tight_layout()
    plot_path = "results/figures/rq3_influence_Unbudgeted.pdf"
    plt.savefig(plot_path, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"[*] Success! Plot saved to {plot_path}")

if __name__ == "__main__":
    run_rq3_influence_distribution()



# ================================ #
# ==== OLD MASTER 3 by 3 PLOT ==== #
# ================================ #


# import os
# import sys
# import numpy as np
# import scipy.sparse as sp
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
# import warnings

# # Mute warnings for clean terminal output
# warnings.filterwarnings('ignore')

# # Ensure Python can find the src module
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# from src.ROMSA.core import (
#     get_polyhedral_target, 
#     get_ellipsoidal_uncorrelated_target,
#     get_ellipsoidal_correlated_target, 
#     get_optimal_stubbornness,
#     build_graph_based_echo_chambers
# )

# def add_trendline_and_stat(ax, x, y, x_scale='linear', y_scale='linear'):
#     """Calculates Pearson correlation and draws a subtle dotted trendline."""
#     # Filter strictly positive values to avoid log(0) errors
#     mask = (x > 0) & (y > 0)
#     x_safe, y_safe = x[mask], y[mask]
    
#     if len(x_safe) < 2: 
#         return
    
#     # Transform data based on the axis scales
#     x_fit = np.log10(x_safe) if x_scale == 'log' else x_safe
#     y_fit = np.log10(y_safe) if y_scale == 'log' else y_safe
    
#     # 1. Calculate and plot the Pearson text box
#     r, _ = pearsonr(x_fit, y_fit)
#     textstr = f'Pearson = {r:.2f}'
#     props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray')
#     ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
#             verticalalignment='top', horizontalalignment='right', bbox=props, zorder=10)
            
#     # 2. Calculate and plot the subtle dotted trendline
#     m, b = np.polyfit(x_fit, y_fit, 1)
#     if x_scale == 'log':
#         x_line = np.geomspace(np.min(x_safe), np.max(x_safe), 100)
#     else:
#         x_line = np.linspace(np.min(x_safe), np.max(x_safe), 100)
        
#     x_line_transformed = np.log10(x_line) if x_scale == 'log' else x_line
#     y_line_transformed = m * x_line_transformed + b
    
#     # Transform back for plotting
#     y_line = 10**y_line_transformed if y_scale == 'log' else y_line_transformed
    
#     # Plot a subtle, dotted, slightly transparent line
#     ax.plot(x_line, y_line, color='black', linewidth=2, linestyle=':', alpha=0.7, zorder=5)

# def run_comprehensive_rq3():
#     print("="*80)
#     print("Generating RQ3: 3x3 Master Plot with Pearson & Dotted Trendlines")
#     print("="*80)
    
#     # Path to Pokec
#     path = "data/processed/W_pokec.npz"
#     if not os.path.exists(path):
#         print(f"Graph not found at {path}!")
#         return

#     W = sp.load_npz(path)
#     n = W.shape[0]
    
#     # 1. Setup Parameters
#     np.random.seed(42)
#     s0 = np.random.uniform(0.0, 1.0, n)
#     rho = np.random.uniform(0.3, 0.7, n)
#     sigma = np.random.uniform(0.3, 0.7, n)
    
#     # Baseline budgets for profiling
#     Gamma_base = 50  
#     Omega_base = 1.1
    
#     print("Calculating PageRank (approx) and Degrees...")
#     pi = np.ones(n) / n
#     for _ in range(50): 
#         pi = W.T @ pi
#     pi = pi / np.sum(pi)
#     degrees = W.getnnz(axis=1)

#     # ---------------------------------------------------------
#     # PART 1: Polyhedral Profiles (Row 1)
#     # ---------------------------------------------------------
#     print("Computing Polyhedral Target & Stubbornness...")
#     q_poly = get_polyhedral_target(s0, Gamma_base, rho)
#     a_poly = get_optimal_stubbornness(W, q_poly)
    
#     mask_poly = q_poly > 1e-8
#     a_act_poly = a_poly[mask_poly]
    
#     # ---------------------------------------------------------
#     # PART 2: Ellipsoidal Uncorrelated Profiles (Row 2)
#     # ---------------------------------------------------------
#     print("Computing Ellipsoidal Uncorrelated Target & Stubbornness...")
#     q_ellip = get_ellipsoidal_uncorrelated_target(s0, Omega_base, sigma)
#     a_ellip = get_optimal_stubbornness(W, q_ellip)
    
#     mask_ellip = q_ellip > 1e-8
#     a_act_ellip = a_ellip[mask_ellip]

#     # ---------------------------------------------------------
#     # PART 3: Sweeps & Echo Chambers (Row 3)
#     # ---------------------------------------------------------
#     print("Sweeping Gamma budget (Polyhedral Diversification)...")
#     gamma_vals = np.geomspace(1, int(n * 0.8), 20) 
#     poly_active_counts = []
#     for g in gamma_vals:
#         q_tmp = get_polyhedral_target(s0, g, rho)
#         poly_active_counts.append(np.sum(q_tmp > 1e-8))
        
#     print("Sweeping Omega budget (Ellipsoidal Diversification)...")
#     omega_vals = np.linspace(0.01, 1.0 / np.max(sigma), 20)
#     ellip_active_counts = []
#     for w in omega_vals:
#         q_tmp = get_ellipsoidal_uncorrelated_target(s0, w, sigma)
#         ellip_active_counts.append(np.sum(q_tmp > 1e-8))

#     print("Building Echo Chambers & Computing Ellipsoidal Shift...")
#     Sigma, Sigma_half = build_graph_based_echo_chambers(W, sigma)
#     q_corr = get_ellipsoidal_correlated_target(s0, Omega_base, Sigma_half)

#     # Calculate Correlation Strength per node for the heatmap
#     if sp.issparse(Sigma):
#         off_diag_Sigma = Sigma - sp.diags(Sigma.diagonal())
#         corr_strength = np.array(np.abs(off_diag_Sigma).sum(axis=1)).flatten()
#     else:
#         off_diag_Sigma = Sigma - np.diag(np.diag(Sigma))
#         corr_strength = np.sum(np.abs(off_diag_Sigma), axis=1)

#     # ========================================== #
#     # 5. Plotting (3x3 Grid)
#     # ========================================== #
#     print("\nGenerating the 9-panel plot...")
#     os.makedirs("results/figures", exist_ok=True)
#     plt.rcParams.update({'font.size': 11, 'axes.linewidth': 1.2})
    
#     fig, axes = plt.subplots(3, 3, figsize=(18, 15), dpi=200)
    
#     # --- ROW 1: Polyhedral Profiles ---
#     color_poly = '#0072B2'
    
#     axes[0, 0].scatter(s0[mask_poly], a_act_poly, color=color_poly, alpha=0.4, s=15)
#     axes[0, 0].set_yscale('log')
#     add_trendline_and_stat(axes[0, 0], s0[mask_poly], a_act_poly, x_scale='linear', y_scale='log')
#     axes[0, 0].set_title("Polyhedral: Stubbornness vs. $s_{0,i}$")
#     axes[0, 0].set_ylabel("$a_i^*$")
    
#     axes[0, 1].scatter(degrees[mask_poly], a_act_poly, color=color_poly, alpha=0.4, s=15)
#     axes[0, 1].set_xscale('log')
#     axes[0, 1].set_yscale('log')
#     add_trendline_and_stat(axes[0, 1], degrees[mask_poly], a_act_poly, x_scale='log', y_scale='log')
#     axes[0, 1].set_title("Polyhedral: Stubbornness vs. Degree")
    
#     axes[0, 2].scatter(rho[mask_poly], a_act_poly, color=color_poly, alpha=0.4, s=15)
#     axes[0, 2].set_yscale('log')
#     add_trendline_and_stat(axes[0, 2], rho[mask_poly], a_act_poly, x_scale='linear', y_scale='log')
#     axes[0, 2].set_title("Polyhedral: Stubbornness vs. $\\rho_i$")

#     # --- ROW 2: Ellipsoidal Profiles ---
#     color_ellip = '#D55E00'
    
#     axes[1, 0].scatter(s0[mask_ellip], a_act_ellip, color=color_ellip, alpha=0.4, s=15)
#     axes[1, 0].set_yscale('log')
#     add_trendline_and_stat(axes[1, 0], s0[mask_ellip], a_act_ellip, x_scale='linear', y_scale='log')
#     axes[1, 0].set_title("Ellipsoidal: Stubbornness vs. $s_{0,i}$")
#     axes[1, 0].set_ylabel("$a_i^*$")
    
#     axes[1, 1].scatter(degrees[mask_ellip], a_act_ellip, color=color_ellip, alpha=0.4, s=15)
#     axes[1, 1].set_xscale('log')
#     axes[1, 1].set_yscale('log')
#     add_trendline_and_stat(axes[1, 1], degrees[mask_ellip], a_act_ellip, x_scale='log', y_scale='log')
#     axes[1, 1].set_title("Ellipsoidal: Stubbornness vs. Degree")
    
#     axes[1, 2].scatter(sigma[mask_ellip], a_act_ellip, color=color_ellip, alpha=0.4, s=15)
#     axes[1, 2].set_yscale('log')
#     add_trendline_and_stat(axes[1, 2], sigma[mask_ellip], a_act_ellip, x_scale='linear', y_scale='log')
#     axes[1, 2].set_title("Ellipsoidal: Stubbornness vs. $\\sigma_i$")

#     # --- ROW 3: Macroscopic Behaviors ---
    
#     # Plot 7: Polyhedral Diversification
#     axes[2, 0].plot(gamma_vals, poly_active_counts, marker='o', color='black', lw=2)
#     axes[2, 0].set_xscale('log')
#     axes[2, 0].set_title("Polyhedral Diversification")
#     axes[2, 0].set_xlabel("Budget ($\\gamma$)")
#     axes[2, 0].set_ylabel("Number of Active Nodes")

#     # Plot 8: Ellipsoidal Diversification
#     axes[2, 1].plot(omega_vals, ellip_active_counts, marker='s', color='black', lw=2)
#     axes[2, 1].set_title("Ellipsoidal Diversification")
#     axes[2, 1].set_xlabel("Budget ($\\omega$)")

#     # Plot 9: Echo Chamber Shift (Uncorr vs Corr)
#     comp_mask = (q_ellip > 1e-8) | (q_corr > 1e-8)
#     scatter = axes[2, 2].scatter(
#         q_ellip[comp_mask], 
#         q_corr[comp_mask], 
#         c=corr_strength[comp_mask],  
#         cmap='coolwarm',             
#         alpha=0.6, 
#         s=15
#     )
    
#     max_val = max(np.max(q_ellip), np.max(q_corr))
#     axes[2, 2].plot([0, max_val], [0, max_val], 'k--', lw=1)
#     axes[2, 2].set_xscale('log')
#     axes[2, 2].set_yscale('log')
#     axes[2, 2].set_title("Echo Chamber Effect")
#     axes[2, 2].set_xlabel("Uncorrelated Target $q_i^*$")
#     axes[2, 2].set_ylabel("Correlated Target $q_i^*$")
    
#     # Add a colorbar
#     cbar = fig.colorbar(scatter, ax=axes[2, 2], fraction=0.046, pad=0.04)
#     cbar.set_label('Correlation Strength', rotation=270, labelpad=15)

#     # Formatting touches
#     for ax in axes.flat:
#         ax.grid(True, linestyle='--', alpha=0.6)

#     plt.tight_layout()
#     plot_path = "results/figures/rq3_3x3_master.pdf"
#     plt.savefig(plot_path, bbox_inches='tight', format='pdf')
#     print(f"Success! Master plot saved to {plot_path}")

# if __name__ == "__main__":
#     run_comprehensive_rq3()


