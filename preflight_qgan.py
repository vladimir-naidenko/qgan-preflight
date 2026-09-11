"""
qgan-preflight: Analytical Pre-flight Diagnostics for Quantum GAN Architectures.
Based on the geometric expressibility bounds and rank barriers (Naidenko, 2026).
"""

from dataclasses import dataclass
from typing import List, Optional
import math
import numpy as np


@dataclass
class TargetState:
    """Target state specifications."""
    m_qubits: int
    spectrum: Optional[np.ndarray] = None  # Eigenvalues in descending order
    purity: Optional[float] = None        # Tr(tau^2) or upper bound p_U

    def __post_init__(self):
        if self.spectrum is not None:
            self.spectrum = np.sort(np.real(self.spectrum))[::-1]
            if not np.isclose(np.sum(self.spectrum), 1.0, atol=1e-3):
                raise ValueError("Spectrum eigenvalues must sum to 1.0.")
            if self.purity is None:
                self.purity = float(np.sum(self.spectrum**2))


@dataclass
class GeneratorArchitecture:
    """Variational generator circuit parameters."""
    m_working_qubits: int
    n_ancillas: int
    n_cross_gates: int
    gate_type: str = "cnot"  # 'cnot', 'cz' (Schmidt rank 2) or 'general' (Schmidt rank <= 4)


@dataclass
class AuditReport:
    passed: bool
    k_epsilon: int
    min_ancillas_needed: int
    min_cross_gates_needed: int
    spectral_barrier_T_K: float
    reasons: List[str]

    def summary(self) -> str:
        status = "\033[92m[PASSED - ARCHITECTURE APPROVED]\033[0m" if self.passed else "\033[91m[FAILED - GEOMETRIC BARRIER DETECTED]\033[0m"
        lines = [
            "=" * 64,
            f" QGAN PRE-FLIGHT DIAGNOSTIC REPORT: {status}",
            "=" * 64,
            f" Effective Spectral Rank k_eps(tau) : {self.k_epsilon}",
            f" Minimum Ancillas Required (a_min)  : {self.min_ancillas_needed}",
            f" Minimum Cross-Gates Required (g_min): {self.min_cross_gates_needed}",
            f" Analytical Lower Bound on Error (r): >= {self.spectral_barrier_T_K:.4f}",
            "-" * 64,
        ]
        if self.passed:
            lines.append(" >> Circuit satisfies necessary rank and entangling constraints.")
            lines.append(" >> Safe to proceed to QPU training stage.")
        else:
            lines.append(" >> CRITICAL VIOLATIONS DETECTED:")
            for r in self.reasons:
                lines.append(f"    * {r}")
            lines.append(" >> RECOMMENDATION: Abort QPU run. Increase ancillas/cross-gates.")
        lines.append("=" * 64)
        return "\n".join(lines)


def run_preflight_audit(
    target: TargetState,
    arch: GeneratorArchitecture,
    epsilon: float
) -> AuditReport:
    """
    Executes the analytical pre-flight test pipeline:
    1. Effective spectral rank audit
    2. Ancilla capacity rule (Corollary 4.5)
    3. Schmidt operator cross-gate barrier (Proposition 4.6)
    4. Purity certificate check (Proposition 5.1)
    """
    reasons = []

    # 1. Determine effective spectral rank k_epsilon
    d = 2**target.m_qubits
    if target.spectrum is not None:
        cum_sum = np.cumsum(target.spectrum)
        k_eps_idx = np.where(cum_sum > 1.0 - epsilon)[0]
        k_eps = int(k_eps_idx[0] + 1) if len(k_eps_idx) > 0 else d
        
        # Exact spectral tail for K = min(2^m, 2^a)
        K_avail = min(d, 2**arch.n_ancillas)
        T_K = float(1.0 - np.sum(target.spectrum[:K_avail]))
    else:
        # Purity lower-bound approximation if spectrum unknown
        p = target.purity if target.purity is not None else 1.0
        # If pure generator (K=1), r >= 1 - sqrt(p)
        k_eps = max(1, math.ceil(1.0 / p))
        K_avail = min(d, 2**arch.n_ancillas)
        T_K = max(0.0, 1.0 - math.sqrt(K_avail * p))

    # 2. Ancilla Rule: a >= ceil(log2(k_eps))
    min_ancillas = math.ceil(math.log2(k_eps)) if k_eps > 1 else 0
    if arch.n_ancillas < min_ancillas:
        reasons.append(
            f"Ancilla Deficit: Provided a = {arch.n_ancillas}, but target requires a >= {min_ancillas}."
        )

    # 3. Cross-register Gate Barrier (Schmidt Rank)
    if k_eps > 1:
        if arch.gate_type.lower() in ["cnot", "cz"]:
            min_gates = math.ceil(math.log2(k_eps))
        else:
            min_gates = math.ceil(math.log2(k_eps) / 2.0)
    else:
        min_gates = 0

    if arch.n_cross_gates < min_gates:
        reasons.append(
            f"Entangling Gate Deficit: Provided g = {arch.n_cross_gates}, but Schmidt rank demands g >= {min_gates}."
        )

    # 4. Error bound check against tolerance
    if T_K >= epsilon:
        reasons.append(
            f"Unreachable Target: Theoretical minimum error r >= {T_K:.4f} exceeds tolerance eps = {epsilon:.4f}."
        )

    passed = len(reasons) == 0
    return AuditReport(
        passed=passed,
        k_epsilon=k_eps,
        min_ancillas_needed=min_ancillas,
        min_cross_gates_needed=min_gates,
        spectral_barrier_T_K=T_K,
        reasons=reasons
    )


if __name__ == "__main__":
    print("\n--- DEMO: 3-Qubit Heisenberg Thermal State (from Section 6.3) ---")
    # Target state from Naidenko (2026), Section 6.3
    heisenberg_spectrum = np.array([0.713, 0.262, 0.018, 0.007, 0.0, 0.0, 0.0, 0.0])
    target = TargetState(m_qubits=3, spectrum=heisenberg_spectrum)
    eps = 0.15

    # Case 1: Pure generator without ancillas (Scheme A)
    arch_pure = GeneratorArchitecture(m_working_qubits=3, n_ancillas=0, n_cross_gates=0)
    report_pure = run_preflight_audit(target, arch_pure, epsilon=eps)
    print(report_pure.summary())

    print("\n")

    # Case 2: Certified architecture with ancillas and CNOTs (Scheme B/C)
    arch_valid = GeneratorArchitecture(m_working_qubits=3, n_ancillas=1, n_cross_gates=2, gate_type="cnot")
    report_valid = run_preflight_audit(target, arch_valid, epsilon=eps)
    print(report_valid.summary())
