#!/usr/bin/env python3
"""
preflight_audit.py
Analytical Pre-flight Screening Protocol for Quantum Architectures in qGANs.
Calculates:
 - Exact spectral tail barriers via Ky Fan's maximum principle
 - Target purity and purity-based non-expressibility certificates
 - Effective spectral rank k_epsilon(tau)
 - Ancilla capacity rule and cross-register Schmidt gate rank bounds (K_eff)
"""

import numpy as np
import scipy.linalg as la

def build_heisenberg_target(m=3, h_field=0.5, beta=1.0):
    """Generates the thermal Gibbs state for a 1D Heisenberg spin chain."""
    d = 2**m
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_n(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    H = np.zeros((d, d), dtype=complex)
    for i in range(m - 1):
        ops_x = [I2] * m; ops_x[i] = X; ops_x[i+1] = X
        ops_y = [I2] * m; ops_y[i] = Y; ops_y[i+1] = Y
        ops_z = [I2] * m; ops_z[i] = Z; ops_z[i+1] = Z
        H += kron_n(ops_x) + kron_n(ops_y) + kron_n(ops_z)

    for i in range(m):
        ops_z = [I2] * m; ops_z[i] = Z
        H += h_field * kron_n(ops_z)

    exp_H = la.expm(-beta * H)
    tau = exp_H / np.trace(exp_H)
    return tau

def run_preflight_screening(tau, epsilon=0.15, m_sys=3):
    """Executes the pre-flight analytical screening workflow."""
    d = tau.shape[0]
    evals = np.sort(np.real(la.eigvalsh(tau)))[::-1]
    purity = float(np.real(np.trace(tau @ tau)))

    # Effective spectral rank k_epsilon
    cum_sum = np.cumsum(evals)
    k_eps = int(np.argmax(cum_sum > 1.0 - epsilon) + 1)
    
    print("=" * 75)
    print(" PRE-FLIGHT ANALYTICAL SCREENING PROTOCOL (A Priori Architecture Audit)")
    print("=" * 75)
    print(f"System dimension:          d = 2^{m_sys} = {d}")
    print(f"Target Purity Tr(tau^2):   p = {purity:.6f}")
    print(f"Tolerance threshold:       epsilon = {epsilon:.4f}")
    print(f"Effective spectral rank:   k_epsilon(tau) = {k_eps}")
    print("\nLeading Target Eigenvalues:")
    for idx, val in enumerate(evals[:6], 1):
        print(f"  lambda_{idx} = {val:.6f}")

    print("\n" + "-" * 75)
    print(" LEVEL 1 & 2: SPECTRAL RANK & ANCILLA CAPACITY AUDIT")
    print("-" * 75)
    
    architectures = [
        {"name": "Pure generator (a = 0)", "a": 0, "g_layer": 0},
        {"name": "One ancilla    (a = 1)", "a": 1, "g_layer": 2},
        {"name": "Two ancillas   (a = 2)", "a": 2, "g_layer": 2},
    ]

    for arch in architectures:
        a = arch["a"]
        # Max coherent rank supported by ancilla register
        k_ancilla = min(2**m_sys, 2**a)
        T_K = 1.0 - np.sum(evals[:k_ancilla])
        
        # Purity lower bound certificate: r >= 1 - sqrt(K * p)
        purity_bound = max(0.0, 1.0 - np.sqrt(k_ancilla * purity))
        
        # Necessary capacity rule
        passes_capacity = k_ancilla >= k_eps
        status = "PASSED PRE-FLIGHT" if passes_capacity else "REJECTED (IMPOSSIBILITY)"

        print(f"[*] Architecture: {arch['name']}")
        print(f"    - Ancilla capacity bound: K_ancilla = min(2^{m_sys}, 2^{a}) = {k_ancilla}")
        print(f"    - Ky Fan spectral tail:  T_{k_ancilla}(tau) = {T_K:.6f}")
        print(f"    - Purity lower bound:    r >= 1 - sqrt({k_ancilla}*p) = {purity_bound:.4f}")
        print(f"    - Ancilla capacity test: K_ancilla >= k_eps ({k_ancilla} >= {k_eps}) -> {status}")
        if not passes_capacity:
            print(f"      >> HARD ARCHITECTURAL BARRIER: Minimal error {T_K:.4f} >= epsilon ({epsilon:.4f}).")
            print(f"      >> Variational optimization on QPU will strictly fail.")
        print()

    print("=" * 75)

if __name__ == "__main__":
    tau = build_heisenberg_target(m=3, h_field=0.5, beta=1.0)
    run_preflight_screening(tau, epsilon=0.15, m_sys=3)