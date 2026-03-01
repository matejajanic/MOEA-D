# MOEA/D: Multiobjective Evolutionary Algorithm based on Decomposition

This project was developed as part of the **Computer Intelligence** course  
at the University of Belgrade – Faculty of Mathematics.

The goal of this project is to implement the **MOEA/D algorithm from scratch**, 
evaluate it on standard benchmark problems, and extend it with custom variations 
and combinatorial applications.

---

## Overview

This repository contains:

- Real-coded MOEA/D implementation
- Binary-coded MOEA/D implementation
- Support for benchmark families:
  - ZDT1–ZDT6
  - ZDT5 (binary)
  - DTLZ2 (M=3, 3D visualization)
- Applications:
  - Feature Selection
  - Project Selection
- Performance evaluation using:
  - IGD (Inverted Generational Distance)
- Experimental comparison:
  - Baseline placeholder variation (v0)
  - SBX + Polynomial Mutation

---

## Implemented Benchmark Problems

### Real-coded

- ZDT1 – Convex Pareto front
- ZDT2 – Non-convex Pareto front
- ZDT3 – Disconnected Pareto front
- ZDT4 – Multimodal problem
- ZDT6 – Non-uniform solution density
- DTLZ2 (M=3) – Spherical Pareto front in 3D

### Binary-coded

- ZDT5
- Feature Selection
- Project Selection

---

## Project Structure

```
.
├── src/
│   ├── moead/          # Core MOEA/D implementation
│   ├── problems/       # Benchmark and application problems
│   └── ...
├── experiments/
│   ├── real/           # Real-coded experiments
│   ├── binary/         # Binary and combinatorial experiments
│   └── common/
├── pyproject.toml
└── README.md
```

---

## Installation

The project uses `pyproject.toml` for dependency management.

### Option 1 – Install in editable mode (recommended)

```bash
python -m venv venv
source venv/bin/activate

pip install -e .
```

### Option 2 – Manual execution with PYTHONPATH

```bash
python -m venv venv
source venv/bin/activate
pip install numpy matplotlib pandas
```

Then set:

```bash
export PYTHONPATH=src
```

---

## Running Experiments

Make sure Python can locate the `src` directory (this should already be satisfied by the installation process,
but if it isn't, run the command below):

```bash
export PYTHONPATH=src
```

### Run ZDT1-ZDT6 (not ZDT5)

```bash
python experiments/real/run_zdt_moead_v0.py --problem zdt[version - 1, 2, 3, 4 or 6]
```
```bash
python experiments/real/run_zdt_random.py --problem zdt[version - 1, 2, 3, 4 or 6]
```
**For comparing the placeholder and SBX + polynomial mutations, run the next command:**
```bash
python experiments/rea/compare_v0_vs_sbx.py --problem zdt[version - 1, 2, 3, 4 or 6]
```

### Run DTLZ2 (M=3)

```bash
python experiments/real/run_dtlz2_m3_3d.py --M 3 --H 10
```

### Run Binary Feature Selection

```bash
python experiments/binary/feature_selection_experiment.py
```
### Run Binary Project Selection

```bash
python experiments/binary/project_selection_experiment.py
```

---

## Performance Evaluation

We evaluate algorithm performance using:

- IGD (Inverted Generational Distance)
- Convergence curves
- Mean ± standard deviation across multiple seeds
- Pareto front visualization (2D and 3D)

---

## Implementation Highlights

- Tchebycheff scalarization
- Simplex-lattice weight generation
- Neighborhood-based update
- SBX crossover (Simulated Binary Crossover)
- Polynomial mutation
- Binary swap mutation for combinatorial problems
- Support for M ≥ 3 objectives

---

## Future Extensions

- Cosine-based neighborhood distance
- Semantic neighborhood for combinatorial problems
- Hypervolume metric implementation
- Text-based multiobjective subset selection problem

---

## References

- Zhang, Q., & Li, H. (2007). MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition.
- Deb, K. et al. (2002). NSGA-II.
- ZDT Benchmark Suite
- DTLZ Benchmark Suite

---

## Authors
- Jovana Brkljac
- Mateja Janic
