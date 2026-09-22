# Expressibility Limits and Pre-flight Diagnostics in qGANs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)

Official Python implementation and numerical benchmark suite for the paper:
> **"Expressibility Limits and Approximate Equilibrium Bounds in Quantum Generative Adversarial Networks"**  
> V. G. Naidenko (*Institute of Mathematics, National Academy of Sciences of Belarus*).

---

## Overview

Quantum generative adversarial networks (qGANs) frequently encounter severe training failures that are commonly attributed to barren plateaus (BP). This repository provides the companion software for a principled **two-tiered pre-flight analytical screening protocol** that identifies insurmountable architectural barriers (rank capacity, ancilla limits, cross-register entangling gate connectivity, and measurement defects) **before committing expensive resources to quantum processing units (QPUs)**.

### Key Theoretical Guarantees Implemented
* **Spectral Tail Rank Barrier (Ky Fan Maximum Principle):**  
  $$\min_{\mathrm{rank}\,\sigma \le k} \Delta(\sigma, \tau) = 1 - \sum_{j=1}^k \lambda_j(\tau) = T_k(\tau)$$
* **Unified Effective Resource Rank:**  
  $$K_{\mathrm{eff}} = \min\left\{2^m, 2^a, \prod_{\ell=1}^g q_\ell\right\}$$
* **Necessary Ancilla Capacity Rule:**  
  $$\min\{2^m, 2^a\} \ge k_\epsilon(\tau) \implies a \ge \lceil\log_2 k_\epsilon(\tau)\rceil$$
* **Purity-Based Impossibility Certificate:**  
  $$r \ge \max\left\{0, 1 - \sqrt{K p}\right\}, \quad p = \operatorname{Tr}(\tau^2)$$

---

## Repository Structure

* `preflight_audit.py`  
  **Zero-cost analytical screening module.** Given target state spectrum invariants or purity bounds, audits candidate ansatzes in milliseconds and outputs formal impossibility certificates without variational optimization.
* `numerical_experiment.py`  
  **Full statistical optimization benchmark.** Replicates the 3-qubit Heisenberg thermal state synthesis study ($N = 30$ independent random seeds across depths $L \in \{2, 3, 4\}$ and ancillas $a \in \{0, 1, 2\}$, 270 total runs). Computes analytical trace-norm subgradients via Helstrom projectors and generates publication figures.
* `qgan_geometric_barrier.pdf` / `.png`  
  High-resolution publication figure generated directly by the benchmark script.
* `requirements.txt`  
  Minimal dependency specifications (pure NumPy/SciPy/Matplotlib, no heavy frameworks required).

---

## Quickstart & Installation

### 1. Clone the repository
```bash
git clone https://github.com/vladimir-naidenko/qgan-preflight.git
cd qgan-preflight
```

### 2. Install dependencies
The code is intentionally written in clean, standard scientific Python without bloated dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Code

### 1. Fast Pre-flight Screening (< 1 second)
To run the a priori architectural audit on the 3-qubit Heisenberg thermal state ($\beta=1.0$, tolerance $\epsilon=0.15$):
```bash
python preflight_audit.py
```
**Output highlights:**
* Rejects pure-state generators ($a=0$) immediately with a hard impossibility certificate: $T_1 \approx 0.2874 \ge \epsilon = 0.15$.
* Certifies that at least $a \ge 1$ ancilla is required to satisfy $K_{\mathrm{eff}} \ge k_\epsilon(\tau) = 2$.

### 2. Full Numerical Optimization & Reproduction (~1-2 minutes)
To reproduce the multi-seed optimization benchmark, generate Table 1 statistics, and render Figure 1:
```bash
python numerical_experiment.py
```
This performs:
1. **Convergence study ($L=3$, $N=30$ seeds):** Median loss trajectories, 10th–90th percentiles, and subgradient evolution.
2. **Depth ablation ($L \in \{2, 3, 4\}$, $N=30$ seeds):** Demonstrates that increasing depth cannot overcome the $a=0$ rank-1 spectral wall ($T_1 \approx 0.2874$).
3. **Automatic export:** Outputs `qgan_geometric_barrier.pdf` and `qgan_geometric_barrier.png`.

---

## Troubleshooting (Windows / Adobe Acrobat)

> **Note for Windows users:**  
> If you have `qgan_geometric_barrier.pdf` open in **Adobe Acrobat Reader**, Windows places an exclusive file lock on it. If you re-run `numerical_experiment.py` while the PDF is open:
> * The script will detect the lock and print a friendly warning instead of crashing.
> * A timestamped copy (e.g., `qgan_geometric_barrier_fallback_<timestamp>.pdf`) will be created automatically.
> * To overwrite the primary file, simply close the PDF tab in Adobe Acrobat before running.

---

## Citation

If you find this framework or code useful in your research, please cite:

```bibtex
@article{Naidenko2026qgan,
  title   = {Expressibility Limits and Approximate Equilibrium Bounds in Quantum Generative Adversarial Networks},
  author  = {Naidenko, Vladimir G.},
  journal = {arXiv preprint arXiv:2603.xxxxx},
  year    = {2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](https://opensource.org/licenses/MIT) file for details.
