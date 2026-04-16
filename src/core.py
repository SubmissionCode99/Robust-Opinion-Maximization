"""
ROMSA: Robust Opinion Maximization via Stubbornness Adjustment, Core Functions

This module contains the highly scalable, O(n log n) and O(|E|) algorithms 
for computing influence targets q^* (Step 1) and finding robust stubbornness values a^* (Step 2)
for solving the ROMSA problem in the Friedkin-Johnsen model.
"""

import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import scipy.linalg as la
from scipy.sparse.linalg import lsmr
from scipy.sparse.linalg import bicgstab, spsolve
from scipy.optimize import minimize_scalar
import pymetis
import warnings

# Mute hardware-level matmul warnings from cvxpy presolvers
warnings.filterwarnings('ignore', message='.*matmul.*')

# ========================================== #
# ===== STEP 1: FIND INFLUENCE TARGET ====== # 
# ========================================== #

def get_nominal_target(s0):
    """Baseline: Allocates all influence to the node with the highest nominal opinion."""
    n = len(s0)
    q_star = np.zeros(n)
    best_node = np.argmax(s0)
    q_star[best_node] = 1.0
    return q_star

# Custom for LP
def get_polyhedral_target(s0, Gamma, rho):
    """Implements Lemma 3: O(n log n) closed-form solution for Polyhedral noise."""
    n = len(s0)
    
    # Category 1: Full-Reliance
    Z_all_in = s0 * (1 - rho)
    best_k_all_in = np.argmax(Z_all_in)
    max_Z_all_in = Z_all_in[best_k_all_in]
    
    # Category 2: Perfect Equalization
    sort_idx = np.argsort(s0)[::-1]
    s0_sorted = s0[sort_idx]
    
    inv_s0 = 1.0 / s0_sorted
    sum_inv_s0 = np.cumsum(inv_s0)
    
    k_values = np.arange(1, n + 1)
    valid_k_mask = k_values > np.floor(Gamma)
    
    Z_eq = np.zeros(n)
    Z_eq[valid_k_mask] = (k_values[valid_k_mask] - Gamma) / sum_inv_s0[valid_k_mask]
    
    best_k_eq_idx = np.argmax(Z_eq)
    max_Z_eq = Z_eq[best_k_eq_idx]
    
    q_star = np.zeros(n)
    
    if max_Z_all_in > max_Z_eq:
        q_star[best_k_all_in] = 1.0
        return q_star
    else:
        k_star = k_values[best_k_eq_idx]
        c_star = 1.0 / sum_inv_s0[best_k_eq_idx]
        
        q_star_sorted = np.zeros(n)
        q_star_sorted[:k_star] = c_star / s0_sorted[:k_star]
        
        q_star[sort_idx] = q_star_sorted
        return q_star

# Custom for Uncorrelated SOCP 
def get_ellipsoidal_uncorrelated_target(s0, Omega, sigma):
    """Implements Lemma 4: O(n log n) closed-form solution for Uncorrelated Ellipsoidal noise."""
    n = len(s0)
    sort_idx = np.argsort(s0)[::-1]
    s0_sorted = s0[sort_idx]
    sigma_sorted = sigma[sort_idx]
    
    inv_var = 1.0 / (sigma_sorted ** 2)
    inv_s0_var = 1.0 / (s0_sorted * sigma_sorted ** 2)
    inv_s02_var = 1.0 / ((s0_sorted * sigma_sorted) ** 2)
    
    A = np.cumsum(inv_s02_var)
    B = np.cumsum(inv_s0_var)
    C = np.cumsum(inv_var) - (Omega ** 2)
    
    discriminant = np.maximum(B**2 - A * C, 0.0) 
    lambda_k = (B - np.sqrt(discriminant)) / A
    
    k_star = None
    lambda_star = None
    s0_padded = np.append(s0_sorted, 0.0)
    
    for k_idx in range(n):
        if s0_padded[k_idx] > lambda_k[k_idx] >= s0_padded[k_idx + 1]:
            k_star = k_idx + 1
            lambda_star = lambda_k[k_idx]
            break
            
    q_star_sorted = np.zeros(n)
    numerator = (s0_sorted[:k_star] - lambda_star) / ((s0_sorted[:k_star] * sigma_sorted[:k_star]) ** 2)
    q_star_sorted[:k_star] = numerator / np.sum(numerator)
    
    q_star = np.zeros(n)
    q_star[sort_idx] = q_star_sorted
    
    return q_star

# Gurobi for LP
def get_polyhedral_target_lp(s0, Gamma, rho, q_init=None, b=None):
    """
    Solves the LP reformulation for Polyhedral noise.
    Supports optional L1 budget constraint on the influence target.
    """
    n = len(s0)
    q = cp.Variable(n, nonneg=True)
    theta = cp.Variable(nonneg=True)
    p = cp.Variable(n, nonneg=True)
    
    objective = cp.Maximize((s0 @ q) - (Gamma * theta) - (rho @ p))
    constraints = [
        theta + p >= cp.multiply(s0, q),
        cp.sum(q) == 1
    ]
    
    # --- ADD INFLUENCE BUDGET CONSTRAINT IF PROVIDED ---
    if q_init is not None and b is not None:
        constraints.append(cp.norm(q - q_init, 1) <= b)
        
    prob = cp.Problem(objective, constraints)
    
    # Gurobi with hyper-strict tolerances, fallback to Clarabel
    try:
        prob.solve(
            solver=cp.GUROBI, 
            verbose=False, 
            FeasibilityTol=1e-9, 
            NumericFocus=3
        )
    except:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Polyhedral LP Solver failed. Status: {prob.status}")
        
    q_star = np.array(q.value).flatten()
    q_star[q_star < 1e-15] = 0.0  # Clean up true floating point artifacts
    sum_q = np.sum(q_star)
    return q_star / sum_q if sum_q > 0 else q_star

# Gurobi for Uncorrelated SOCP
def get_ellipsoidal_uncorrelated_target_socp(s0, Omega, sigma, q_init=None, b=None):
    """
    Solves the SOCP reformulation for Uncorrelated Ellipsoidal noise.
    Supports optional L1 budget constraint on the influence target.
    """
    n = len(s0)
    q = cp.Variable(n, nonneg=True)
    t = cp.Variable(nonneg=True)
    
    scaled_q = cp.multiply(sigma * s0, q)
    
    objective = cp.Maximize((s0 @ q) - (Omega * t))
    constraints = [
        cp.norm(scaled_q, 2) <= t,
        cp.sum(q) == 1
    ]
    
    # --- ADD INFLUENCE BUDGET CONSTRAINT IF PROVIDED ---
    if q_init is not None and b is not None:
        constraints.append(cp.norm(q - q_init, 1) <= b)
        
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(
            solver=cp.GUROBI, 
            verbose=False, 
            FeasibilityTol=1e-9, 
            NumericFocus=3
        )
    except:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Uncorrelated SOCP Solver failed. Status: {prob.status}")
        
    q_star = np.array(q.value).flatten()
    q_star[q_star < 1e-15] = 0.0  # Clean up true floating point artifacts
    sum_q = np.sum(q_star)
    return q_star / sum_q if sum_q > 0 else q_star

# Gurobi for Correlated SOCP
def get_ellipsoidal_correlated_target(s0, Omega, Sigma_half, q_init=None, b=None):
    """
    Solves the SOCP reformulation for Correlated Ellipsoidal noise.
    Supports optional L1 budget constraint on the influence target.
    """
    n = len(s0)
    q = cp.Variable(n, nonneg=True)
    t = cp.Variable(nonneg=True)
    
    scaled_q = cp.multiply(s0, q)
    objective = cp.Maximize((s0 @ q) - (Omega * t))
    constraints = [
        cp.sum(q) == 1,
        cp.norm(Sigma_half @ scaled_q, 2) <= t
    ]
    
    # --- ADD INFLUENCE BUDGET CONSTRAINT IF PROVIDED ---
    if q_init is not None and b is not None:
        constraints.append(cp.norm(q - q_init, 1) <= b)
        
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(
            solver=cp.GUROBI, 
            verbose=False, 
            FeasibilityTol=1e-9, 
            NumericFocus=3
        )
    except:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Correlated SOCP Solver failed. Status: {prob.status}")
        
    q_star = np.array(q.value).flatten()
    q_star[q_star < 1e-15] = 0.0  # Clean up true floating point artifacts
    sum_q = np.sum(q_star)
    return q_star / sum_q if sum_q > 0 else q_star

# ======================================== #
# ===== STEP 2: FIND STUBBORNNESS  ======= #
# ======================================== #

def get_optimal_stubbornness(W, q_star, a_init=None, tol=1e-12):
    """
    Implements Theorem 1: Constructs physical stubbornness a*.
    If a_init is provided, it performs a 3-phase deterministic search 
    to find the optimal eta* that minimizes the L2 distance to a_init.
    """
    n = W.shape[0]
    
    # Power Iteration for PageRank
    pi = np.ones(n) / n
    for _ in range(5000):
        pi_next = W.T @ pi
        if np.max(np.abs(pi_next - pi)) < tol:
            pi = pi_next
            break
        pi = pi_next
    pi = pi / np.sum(pi)  
    
    I = sp.eye(n, format='csr')
    C = I - W.T
    b = (np.ones(n) / n) - (W.T @ q_star)
    
    # LSMR Solver
    y0 = lsmr(C, b, atol=1e-6, btol=1e-6, maxiter=10000)[0]
    
    active = q_star > 1e-12
    safe_pi = np.maximum(pi[active], 1e-15) 
    
    # Calculate absolute minimum bound for eta
    eta_min = np.max((q_star[active] - y0[active]) / safe_pi)
    
    # --------------------------------------------------- #
    # --- NO BUDGET: Default to eta_min Intervention ---- #
    if a_init is None:
        optimal_eta = eta_min
        
    # --- BUDGET APPLIED: 3-Phase Deterministic ETA Search --- #
    else:
        # We need a fast cost function for the active nodes
        def cost_func(eta):
            a_curr = q_star[active] / (y0[active] + eta * safe_pi)
            return np.sum((a_curr - a_init[active])**2)

        # Phase 1: Domain Bounding
        # Find eta_max such that maximum a_i* becomes negligibly small (e.g., < 1e-4)
        epsilon = 1e-4
        eta_max = np.max((q_star[active] / epsilon - y0[active]) / safe_pi)
        
        # Safety catch in case of extreme graphs
        if eta_max <= eta_min:
            eta_max = eta_min + 1.0
            
        # Phase 2: Global Grid Search (O(n) cost)
        K = 10000
        eta_grid = np.linspace(eta_min, eta_max, K)
        costs = np.zeros(K)
        
        # Fast loop to evaluate all grid points without blowing up memory
        for idx, e in enumerate(eta_grid):
            costs[idx] = cost_func(e)
            
        best_idx = np.argmin(costs)
        
        # Phase 3: Local Polish (Parabolic Interpolation via Brent's Method)
        # Create a microscopic bracket using the neighbors of the best grid point
        left_idx = max(0, best_idx - 1)
        right_idx = min(K - 1, best_idx + 1)
        
        bound_left = eta_grid[left_idx]
        bound_right = eta_grid[right_idx]
        
        if bound_left == bound_right:
            optimal_eta = bound_left
        else:
            # Brent's method guarantees exact continuous minimum within the convex bracket
            res = minimize_scalar(cost_func, bounds=(bound_left, bound_right), method='bounded')
            optimal_eta = res.x
    # ------------------------------------------------------------- #
    
    # Construct final a_star using the selected eta
    a_star = np.zeros(n)
    a_star[active] = q_star[active] / (y0[active] + optimal_eta * safe_pi)
    
    return np.clip(a_star, 0.0, 1.0)


# ============================================ #
# ===== WORST-CASE NOISE, See Appendix B ===== #
# ============================================ #

def get_worst_case_polyhedral_opinions(q, s0, Gamma, rho):
    """
    Simulates the adversary's optimal attack under polyhedral noise.
    Solves the continuous knapsack problem to find the worst-case "noisy" opinions.
    """
    n = len(s0)
    c = q * s0  # Damage coefficients (contribution)
    
    # Sort nodes by damage coefficient descending
    sort_idx = np.argsort(c)[::-1]
    rho_sorted = rho[sort_idx]
    
    # Vectorized continuous knapsack
    cum_rho = np.cumsum(rho_sorted)
    delta_sorted = np.zeros(n)
    
    # Find nodes where we can apply maximum damage (rho_i)
    full_take_mask = cum_rho <= Gamma
    delta_sorted[full_take_mask] = rho_sorted[full_take_mask]
    
    # Handle the fractional node where the budget runs out
    if not np.all(full_take_mask):
        fractional_idx = np.argmin(full_take_mask) # First index where it's False
        remaining_budget = Gamma - (cum_rho[fractional_idx] - rho_sorted[fractional_idx])
        if remaining_budget > 0:
            delta_sorted[fractional_idx] = remaining_budget
            
    # Un-sort delta to match original node indices
    delta = np.zeros(n)
    delta[sort_idx] = delta_sorted
    
    # Return the worst-case noisy opinions
    return s0 * (1 - delta)

def get_worst_case_ellipsoidal_opinions(q, s0, Omega, Sigma):
    """
    Simulates the adversary's optimal attack under ellipsoidal noise.
    Uses the gradient alignment closed-form solution.
    Sigma is expected to be a scipy.sparse matrix or a 1D array for diagonal matrices.
    """
    c = q * s0
    
    if sp.issparse(Sigma):
        Sigma_c = Sigma @ c
        c_Sigma_c = c.T @ Sigma_c
    else:
        # If diagonal Sigma is passed as a 1D array
        Sigma_c = Sigma * c
        c_Sigma_c = np.sum(c * Sigma_c)
        
    # Prevent division by zero if influence vector is entirely 0
    if c_Sigma_c < 1e-16:
        delta = np.zeros_like(s0)
    else:
        delta = Omega * (Sigma_c / np.sqrt(c_Sigma_c))
        
    return s0 * (1 - delta)


# ========================= #
# === EVALUATION HELPER === #
# ========================= # 

def evaluate_influence(W, a_vector):
    """
    Evaluates the exact physical steady-state of the network.
    Uses LSMR instead of simple iterations.
    """
    n = W.shape[0]
    I = sp.eye(n, format='csr')
    A_sparse = sp.diags(a_vector)
    
    # Formulate the exact equilibrium system: (I - W^T(I - A)) x = b
    C_eval = I - W.T @ (I - A_sparse)
    b_val = np.ones(n) / n
    
    # LSMR reliably computes the steady state
    x_val = lsmr(C_eval, b_val, atol=1e-12, btol=1e-12, maxiter=100000)[0]
    
    q_simulated = a_vector * x_val
    return q_simulated

## Find q with iteration = slower and not as reliable
# def evaluate_influence(W, a_vector, tol=1e-8, max_iter=100000):
#     """
#     Simulates the actual physical Friedkin-Johnsen influence vector dynamics by iterations (OLD).
#     """
#     n = W.shape[0]
#     I = sp.eye(n, format='csr')
#     A_sparse = sp.diags(a_vector)
    
#     W_T_I_A = W.T @ (I - A_sparse)
#     b_val = np.ones(n) / n  # Correct 1/n formulation
    
#     x_val = b_val.copy()
#     for i in range(max_iter):
#         x_next = W_T_I_A @ x_val + b_val
#         if np.max(np.abs(x_next - x_val)) < tol:
#             x_val = x_next
#             break
            
#     q_simulated = a_vector * x_val
#     return q_simulated


# ============================================= #
# === BUILD CORRELATION MATRIX Σ WITH METIS === #
# ============================================= # 

def build_graph_based_echo_chambers(W, sigma, target_size=15, corr=0.4):
    """
    Builds Non-Diagonal Correlated Sigma and Sigma^(1/2) using graph topology.
    Uses the METIS graph partitioning algorithm to group nodes into optimal, 
    strongly connected echo chambers (minimizing edge cuts).
    """
    n = W.shape[0]
    
    # 1. Determine the number of partitions (echo chambers)
    n_parts = max(1, n // target_size)
    
    # 2. METIS requires an undirected graph for optimal partitioning.
    # We symmetrize the interaction matrix W.
    W_sym = W + W.T
    W_sym = W_sym.tocsr()
    
    # 3. Run METIS Partitioning
    # pymetis is extremely fast and takes the CSR representation directly
    n_cuts, membership = pymetis.part_graph(nparts=n_parts, xadj=W_sym.indptr, adjncy=W_sym.indices)
    
    # 4. Group nodes by their METIS partition ID
    clusters = {}
    for node_idx, part_id in enumerate(membership):
        if part_id not in clusters:
            clusters[part_id] = []
        clusters[part_id].append(node_idx)
        
    row_idx, col_idx = [], []
    val_Sigma, val_Sigma_half = [], []
    
    clusters_formed = len(clusters)
    
    # 5. Build the Block-Diagonal Covariance Matrices
    for part_id, cluster in clusters.items():
        k = len(cluster)
        if k == 0:
            continue
            
        sig = sigma[cluster]
        
        block_S = np.zeros((k, k))
        for r in range(k):
            for c in range(k):
                if r == c:
                    block_S[r, c] = sig[r]**2
                else:
                    block_S[r, c] = corr * sig[r] * sig[c]
                    
        # Compute matrix square root for the community block
        block_S_half = np.real(la.sqrtm(block_S))
        
        for r in range(k):
            for c in range(k):
                row_idx.append(cluster[r])
                col_idx.append(cluster[c])
                val_Sigma.append(block_S[r, c])
                val_Sigma_half.append(block_S_half[r, c])

    # 6. Reconstruct the global sparse matrices
    Sigma = sp.coo_matrix((val_Sigma, (row_idx, col_idx)), shape=(n, n)).tocsr()
    Sigma_half = sp.coo_matrix((val_Sigma_half, (row_idx, col_idx)), shape=(n, n)).tocsr()
    
    return Sigma, Sigma_half