#!/usr/bin/env python3
"""
numerical_experiment.py
Rigorous numerical evaluation of expressibility limits in qGANs.
Features:
 - Multi-seed statistical benchmark (L = 3, N = 30 seeds)
 - Full depth-ablation study (L in {2, 3, 4}, N = 30 seeds strictly)
 - Post-update terminal evaluation (true state after 180 Adam updates)
 - Exact trace-norm analytical subgradient calculus
 - Generates publication figure: 'qgan_geometric_barrier3.pdf' and '.png'
 - Prints full statistics table for direct LaTeX synchronization
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time

# ==============================================================================
# 1. PHYSICAL TARGET: 1D 3-Qubit Heisenberg Spin Chain (m = 3, d = 8)
# ==============================================================================
m = 3
d = 2**m

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_n(op_list):
    res = op_list[0]
    for op in op_list[1:]:
        res = np.kron(res, op)
    return res

H = np.zeros((d, d), dtype=complex)
h_field = 0.5

for i in range(m - 1):
    ops_x = [I2] * m; ops_x[i] = X; ops_x[i+1] = X
    ops_y = [I2] * m; ops_y[i] = Y; ops_y[i+1] = Y
    ops_z = [I2] * m; ops_z[i] = Z; ops_z[i+1] = Z
    H += kron_n(ops_x) + kron_n(ops_y) + kron_n(ops_z)

for i in range(m):
    ops_z = [I2] * m; ops_z[i] = Z
    H += h_field * kron_n(ops_z)

beta = 1.0
exp_H = la.expm(-beta * H)
tau = exp_H / np.trace(exp_H)

lambdas = np.sort(np.real(la.eigvalsh(tau)))[::-1]
T1 = 1.0 - lambdas[0]
T2 = 1.0 - np.sum(lambdas[:2])
T4 = 1.0 - np.sum(lambdas[:4])

# ==============================================================================
# 2. VARIATIONAL HEA GENERATOR & EXACT ANALYTICAL SUBGRADIENTS
# ==============================================================================
def ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)

def d_ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return 0.5 * np.array([[-s, -c], [c, -s]], dtype=complex)

def cnot(c, t, n):
    dim = 2**n
    res = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[c] == 1:
            bits[t] ^= 1
        new_i = sum(b << (n - 1 - k) for k, b in enumerate(bits))
        res[new_i, i] = 1.0
    return res

def precompute_cnots(n):
    return [cnot(i, (i + 1) % n, n) for i in range(n)]

def partial_trace(rho_full, n_sys, n_anc):
    if n_anc == 0:
        return rho_full
    d_sys = 2**n_sys
    d_anc = 2**n_anc
    reshaped = rho_full.reshape((d_sys, d_anc, d_sys, d_anc))
    return np.trace(reshaped, axis1=1, axis2=3)

def trace_distance_and_projector(sigma, target):
    diff = target - sigma
    diff_herm = (diff + diff.conj().T) / 2.0
    evals, evecs = la.eigh(diff_herm)
    
    dist = 0.5 * np.sum(np.abs(evals))
    pos_mask = evals > 1e-12
    if np.any(pos_mask):
        P_plus = evecs[:, pos_mask] @ evecs[:, pos_mask].conj().T
    else:
        P_plus = np.zeros_like(diff)
    return dist, P_plus

def simulate_circuit_and_gradient(params, n, layers, cnot_list, target_tau, m_sys, a_anc):
    num_params = (layers + 1) * n
    dim = 2**n
    
    ry_mats = []
    d_ry_mats = []
    idx = 0
    for l in range(layers + 1):
        layer_ry = [ry(params[idx + i]) for i in range(n)]
        layer_dry = [d_ry(params[idx + i]) for i in range(n)]
        ry_mats.append(kron_n(layer_ry))
        d_ry_mats.append(layer_dry)
        idx += n
        
    U_steps = []
    current_U = np.eye(dim, dtype=complex)
    for l in range(layers):
        current_U = ry_mats[l] @ current_U
        U_steps.append(('ry', l, current_U))
        for c_gate in cnot_list:
            current_U = c_gate @ current_U
        U_steps.append(('cnot', l, current_U))
    current_U = ry_mats[layers] @ current_U
    U_steps.append(('ry', layers, current_U))
    
    U_total = current_U
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1.0
    psi = U_total @ psi0
    
    rho = np.outer(psi, psi.conj())
    sigma = partial_trace(rho, m_sys, a_anc)
    loss, P_plus = trace_distance_and_projector(sigma, target_tau)
    
    if a_anc > 0:
        P_full = np.kron(P_plus, np.eye(2**a_anc, dtype=complex))
    else:
        P_full = P_plus
        
    grad = np.zeros(num_params)
    for l in range(layers + 1):
        if l == 0:
            U_before = np.eye(dim, dtype=complex)
        else:
            U_before = U_steps[2 * (l - 1) + 1][2]
            
        U_after = U_total @ la.inv(ry_mats[l] @ U_before)
        psi_before = U_before @ psi0
        
        for q in range(n):
            p_idx = l * n + q
            ops_dry = [ry(params[l * n + i]) for i in range(n)]
            ops_dry[q] = d_ry_mats[l][q]
            d_kron = kron_n(ops_dry)
            d_psi = U_after @ (d_kron @ psi_before)
            grad[p_idx] = -2.0 * np.real(psi.conj().T @ P_full @ d_psi)
            
    return loss, grad

# ==============================================================================
# 3. EXPERIMENT CONFIGURATION (N = 30 everywhere for exact replication)
# ==============================================================================
scenarios = [
    {"a": 0, "label": "Pure generator (a = 0)", "color": "#d62728", "bound": T1},
    {"a": 1, "label": "One ancilla (a = 1)",    "color": "#ff7f0e", "bound": T2},
    {"a": 2, "label": "Two ancillas (a = 2)",   "color": "#2ca02c", "bound": T4}
]

N_SEEDS = 30
STEPS = 160
LR = 0.08
L_FIXED = 3

depths = [2, 3, 4]
N_SEEDS_DEPTH = 30
STEPS_DEPTH = 180

print("=" * 75)
print(f" RUNNING BENCHMARKS: Convergence (N={N_SEEDS}) & Depth Ablation (N={N_SEEDS_DEPTH})")
print("=" * 75)

start_time = time.time()

# 3.1 Convergence benchmark (L = 3)
stats_losses = {sc["a"]: np.zeros((N_SEEDS, STEPS)) for sc in scenarios}
stats_grads = {sc["a"]: np.zeros((N_SEEDS, STEPS)) for sc in scenarios}

for sc in scenarios:
    a = sc["a"]
    n_total = m + a
    cnot_list = precompute_cnots(n_total)
    num_params = (L_FIXED + 1) * n_total
    print(f"[*] Benchmark L={L_FIXED}, a={a} ({N_SEEDS} seeds)...")
    
    for s in range(N_SEEDS):
        np.random.seed(1000 + s * 37 + a * 7)
        params = np.random.normal(0, 0.1, num_params)
        m_adam = np.zeros(num_params)
        v_adam = np.zeros(num_params)
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        
        for t in range(1, STEPS + 1):
            loss_val, grad = simulate_circuit_and_gradient(
                params, n_total, L_FIXED, cnot_list, tau, m, a
            )
            stats_losses[a][s, t - 1] = loss_val
            stats_grads[a][s, t - 1] = np.linalg.norm(grad)
            
            m_adam = beta1 * m_adam + (1 - beta1) * grad
            v_adam = beta2 * v_adam + (1 - beta2) * (grad**2)
            m_hat = m_adam / (1 - beta1**t)
            v_hat = v_adam / (1 - beta2**t)
            params -= LR * m_hat / (np.sqrt(v_hat) + eps_adam)

# 3.2 Depth ablation benchmark (L in {2, 3, 4}, N = 30)
print(f"\n[*] Running Depth Ablation L in {depths} with N={N_SEEDS_DEPTH} seeds...")
depth_final_loss = {sc["a"]: {L: [] for L in depths} for sc in scenarios}

for sc in scenarios:
    a = sc["a"]
    n_total = m + a
    cnot_list = precompute_cnots(n_total)
    
    for L in depths:
        num_params = (L + 1) * n_total
        print(f"    - Running a={a}, L={L} ({num_params} params, {N_SEEDS_DEPTH} seeds)...")
        for s in range(N_SEEDS_DEPTH):
            np.random.seed(2000 + s * 43 + a * 13 + L * 3)
            params = np.random.normal(0, 0.1, num_params)
            m_adam = np.zeros(num_params)
            v_adam = np.zeros(num_params)
            beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
            
            for t in range(1, STEPS_DEPTH + 1):
                loss_val, grad = simulate_circuit_and_gradient(
                    params, n_total, L, cnot_list, tau, m, a
                )
                m_adam = beta1 * m_adam + (1 - beta1) * grad
                v_adam = beta2 * v_adam + (1 - beta2) * (grad**2)
                m_hat = m_adam / (1 - beta1**t)
                v_hat = v_adam / (1 - beta2**t)
                params -= LR * m_hat / (np.sqrt(v_hat) + eps_adam)
            
            # Post-update evaluation: evaluates the true state after 180 updates
            final_loss, _ = simulate_circuit_and_gradient(
                params, n_total, L, cnot_list, tau, m, a
            )
            depth_final_loss[a][L].append(final_loss)

total_elapsed = time.time() - start_time
print(f"\nAll computations completed successfully in {total_elapsed:.1f} seconds.")

# ==============================================================================
# 4. PRINT FORMATTED SUMMARY TABLE (Direct input for Table 1 in paper)
# ==============================================================================
print("\n" + "=" * 95)
print(" SUMMARY TABLE OF RESULTS (N = 30 seeds, 180 iterations, Post-Update)")
print("=" * 95)
header = f"{'Ancillas':<10} {'Depth':<8} {'Params':<8} {'Final Error (Mean +- SD)':<26} {'Median':<10} {'Best Run':<10} {'Barrier T_K':<12} {'Tol. eps=0.15'}"
print(header)
print("-" * 95)

for sc in scenarios:
    a = sc["a"]
    n_total = m + a
    barrier_str = f"T_{2**a if a>0 else 1} ~ {sc['bound']:.4f}"
    for L in depths:
        num_params = (L + 1) * n_total
        res = depth_final_loss[a][L]
        mean_val = np.mean(res)
        std_val = np.std(res)
        med_val = np.median(res)
        best_val = np.min(res)
        tol_status = "Pass" if med_val < 0.15 else "Fail"
        mean_str = f"{mean_val:.4f} +- {std_val:.4f}"
        print(f"a = {a:<6} L = {L:<4} {num_params:<8} {mean_str:<26} {med_val:<10.4f} {best_val:<10.4f} {barrier_str:<12} {tol_status}")

print("=" * 95)

# ==============================================================================
# 5. PUBLICATION-QUALITY FIGURE (Saved as 'qgan_geometric_barrier3.pdf')
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(12.0, 9.5), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

x_steps = np.arange(1, STEPS + 1)

# Panel (a)
for sc in scenarios:
    a = sc["a"]
    data = stats_losses[a]
    median = np.median(data, axis=0)
    q10 = np.percentile(data, 10, axis=0)
    q90 = np.percentile(data, 90, axis=0)
    
    ax1.plot(x_steps, median, label=f"{sc['label']} (median)", color=sc["color"], lw=2.2)
    ax1.fill_between(x_steps, q10, q90, color=sc["color"], alpha=0.18, label=f"{sc['label']} (10-90%)")
    ax1.axhline(sc["bound"], color=sc["color"], linestyle="--", lw=1.6, alpha=0.85,
                label=rf"Barrier $T_{{{2**a if a>0 else 1}}} = {sc['bound']:.4f}$")

ax1.axhline(0.15, color="black", linestyle=":", lw=1.8, label=r"Tolerance $\epsilon = 0.15$")
ax1.set_ylabel(r"Trace distance $\Delta(\sigma_\theta, \tau)$", fontsize=11, fontweight="bold")
ax1.set_xlabel("Optimization step", fontsize=11, fontweight="bold")
ax1.set_title(r"(a) Multi-seed convergence trajectories ($N=30$) vs. exact spectral barriers",
              fontsize=12, fontweight="bold", pad=8)
ax1.set_ylim(-0.01, 0.75)
ax1.set_xlim(1, STEPS)
ax1.legend(loc="upper right", frameon=True, fontsize=8.0, ncol=3)

# Panel (b)
width = 0.22
x_indices = np.arange(len(depths))

for idx, sc in enumerate(scenarios):
    a = sc["a"]
    means = [np.mean(depth_final_loss[a][L]) for L in depths]
    stds = [np.std(depth_final_loss[a][L]) for L in depths]
    mins = [np.min(depth_final_loss[a][L]) for L in depths]
    
    ax2.bar(x_indices + (idx - 1) * width, means, width, yerr=stds, capsize=4,
            label=sc["label"], color=sc["color"], alpha=0.85, edgecolor="black", linewidth=0.8)
    ax2.scatter(x_indices + (idx - 1) * width, mins, color="black", s=25, zorder=5,
                marker="d", label="Best run" if idx == 0 else "")

for sc in scenarios:
    ax2.axhline(sc["bound"], color=sc["color"], linestyle="--", lw=1.2, alpha=0.7)

ax2.axhline(0.15, color="black", linestyle=":", lw=1.5)
ax2.set_xticks(x_indices)
ax2.set_xticklabels([f"L = {L}" for L in depths], fontsize=10, fontweight="bold")
ax2.set_ylabel(r"Final trace distance $\Delta(\sigma_*, \tau)$", fontsize=11, fontweight="bold")
ax2.set_xlabel("Ansatz circuit depth $L$", fontsize=11, fontweight="bold")
ax2.set_title(r"(b) Expressibility vs. depth ablation ($N=30$ seeds)", fontsize=12, fontweight="bold", pad=8)
ax2.set_ylim(0, 0.40)
ax2.legend(loc="upper right", frameon=True, fontsize=8.0)

# Panel (c)
for sc in scenarios:
    a = sc["a"]
    data_grad = stats_grads[a]
    median_grad = np.median(data_grad, axis=0)
    q10_g = np.percentile(data_grad, 10, axis=0)
    q90_g = np.percentile(data_grad, 90, axis=0)
    
    ax3.plot(x_steps, median_grad, label=sc["label"], color=sc["color"], lw=1.8)
    ax3.fill_between(x_steps, q10_g, q90_g, color=sc["color"], alpha=0.15)

ax3.set_xlabel("Optimization step", fontsize=11, fontweight="bold")
ax3.set_ylabel(r"Exact subgradient norm $\|\nabla_\theta \Delta\|$", fontsize=11, fontweight="bold")
ax3.set_title(r"(c) Evolution of analytical subgradient norm", fontsize=12, fontweight="bold", pad=8)
ax3.set_yscale("log")
ax3.set_xlim(1, STEPS)
ax3.legend(loc="upper right", frameon=True, fontsize=8.5)

plt.tight_layout()
figure_filename = "qgan_geometric_barrier3"
plt.savefig(f"{figure_filename}.pdf", bbox_inches="tight")
plt.savefig(f"{figure_filename}.png", bbox_inches="tight")
print(f"\n[OK] Generated high-resolution publication figures: '{figure_filename}.pdf' and '.png'!")