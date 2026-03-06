# MOEA/D: Multiobjective Evolutionary Algorithm based on Decomposition

This project was developed as part of the **Computer Intelligence** course  
at the **University of Belgrade – Faculty of Mathematics**.

The goal of the project is to implement the **MOEA/D algorithm from scratch**, evaluate it on standard benchmark problems, and apply it to several **real-world multiobjective optimization tasks**.

All components of the algorithm were **implemented manually without using existing evolutionary optimization libraries**.

---

# Overview

The repository contains:

- Full implementation of the **MOEA/D algorithm**
- Support for **real-coded and binary-coded representations**
- Evaluation on **benchmark multiobjective optimization problems**
- Applications to **real-world combinatorial optimization tasks**
- Automated **experiment runner**
- Performance evaluation using **IGD and Hypervolume metrics**
- Visualization of Pareto fronts and optimization dynamics

---

# Implemented Problems

## Benchmark Problems

Real-coded benchmark functions:

- ZDT1
- ZDT2
- ZDT3
- ZDT4
- ZDT6
- DTLZ2 (3 objectives)

Binary-coded benchmark:

- ZDT5

---

## Real-world Multiobjective Problems

Binary combinatorial problems implemented in the project:

- **Feature Selection**
- **Project Selection with Budget Constraint**
- **Text Summarization (multiobjective sentence selection)**

---

# Algorithm Components

The implemented MOEA/D algorithm includes:

- **Tchebycheff scalarization**
- **Simplex-lattice weight generation**
- **Neighborhood-based subproblem interaction**
- **SBX crossover (Simulated Binary Crossover)**
- **Polynomial mutation**
- **Binary mutation operators**
- Support for **multiobjective problems with M ≥ 3 objectives**

---

# Project Structure

```text
docs/                           # Project documentation

src/
├── experiments/                
│   ├── benchmarks/            
│   ├── comparisons/           
│   ├── real_world/             
│   ├── __init__.py
│   ├── common.py               
│   ├── plotting.py             
│   └── run.py                  # Main experiment runner (command-line interface)
│
├── moead/                      # Core MOEA/D algorithm implementation
│   ├── __init__.py
│   ├── algorithm.py            # Main MOEA/D optimization loop
│   ├── metrics.py              # Performance metrics (IGD, Hypervolume)
│   ├── scalarizing.py          # Scalarization methods (Tchebycheff)
│   ├── variation.py            # Real-coded variation operators (SBX, mutation)
│   ├── variation_binary.py     # Binary variation operators
│   └── weights.py              # Weight vector generation and neighborhood structure
│
└── problems/                   # Definitions of optimization problems
    ├── __init__.py
    ├── dtlz2.py                
    ├── feature_selection.py    
    ├── project_selection.py    
    ├── text_summarization.py   
    ├── zdt.py                  
    └── zdt5.py                
```

# Installation

The project uses **pyproject.toml** with **setuptools** for dependency management.

Create a virtual environment and install the project in editable mode:

```bash
python -m venv venv
source venv/bin/activate

pip install -e .
```

This will:

- install required dependencies
- make the `src` directory available as a Python package
- allow running experiments directly from the repository

Dependencies specified in `pyproject.toml` include:

- numpy
- matplotlib
- scikit-learn

---

# Alternative Installation (Manual)

If editable installation is not used, dependencies can be installed manually:

```bash
pip install numpy matplotlib scikit-learn
```

and the source directory must be added to the Python path:

```bash
export PYTHONPATH=src
```

---

# Running

All experiments are executed through the **main experiment runner**:

```
python src/experiments/run.py
```

The runner supports multiple experiment modes and configurable algorithm parameters.

---

## Basic Usage

Minimal example:

```bash
python src/experiments/run.py --suite benchmark --problem zdt1
```

This runs the **MOEA/D algorithm on the ZDT1 benchmark problem** using default parameters.

---

## Experiment Suites

The runner supports three main experiment modes:

```
--suite benchmark
--suite realworld
--suite compare
```

---

## Benchmark Suite

Standard multiobjective benchmark problems.

Supported problems:

```
zdt1
zdt2
zdt3
zdt4
zdt5
zdt6
dtlz2
```

Example:

```bash
python src/experiments/run.py \
    --suite benchmark \
    --problem zdt3
```

---

## Real-world Problems

Multiobjective combinatorial optimization problems implemented in the project.

Supported problems:

```
feature
project
textsum
```

Example:

```bash
python src/experiments/run.py \
    --suite realworld \
    --problem feature
```

---

## Algorithm Comparison

Runs experiments for **algorithm variant comparison** and **performance statistics across multiple seeds**.

Example:

```bash
python src/experiments/run.py \
    --suite compare \
    --problem zdt1
```

This mode compares:

- placeholder variation operator
- SBX crossover + polynomial mutation

Metrics reported:

- IGD
- Hypervolume

---

## MOEA/D Parameters

The following parameters control the MOEA/D algorithm:

| Parameter | Description | Default |
|----------|-------------|--------|
| `--pop_size` | population size | 200 |
| `--T` | neighborhood size | 20 |
| `--n_gen` | number of generations | 200 |
| `--seed` | random seed | 42 |
| `--nr` | max replacements per update | 2 |

Example:

```bash
python src/experiments/run.py \
    --suite benchmark \
    --problem zdt1 \
    --pop_size 300 \
    --n_gen 500
```

---

## Variation Operators

The variation operator can be selected with:

```
--variation
```

Options:

```
sbx
placeholder
```

Example:

```bash
python src/experiments/run.py \
    --suite benchmark \
    --problem zdt1 \
    --variation sbx
```

---

## Crossover and Mutation Parameters

For real-coded problems the following parameters can be configured:

| Parameter | Description | Default |
|----------|-------------|--------|
| `--eta_c` | SBX crossover distribution index | 20 |
| `--eta_m` | polynomial mutation distribution index | 20 |
| `--p_c` | crossover probability | 1.0 |
| `--p_m` | mutation probability | auto |

Example:

```bash
python src/experiments/run.py \
    --suite benchmark \
    --problem zdt2 \
    --eta_c 15 \
    --eta_m 25
```

---

## Problem-Specific Parameters

Some problems require additional arguments.

### DTLZ2

```
--M
--n_var
```

Example:

```bash
python src/experiments/run.py \
    --suite benchmark \
    --problem dtlz2 \
    --M 3 \
    --n_var 30
```

---

### Project Selection

```
--n_projects
--budget_ratio
```

Example:

```bash
python src/experiments/run.py \
    --suite realworld \
    --problem project \
    --n_projects 200 \
    --budget_ratio 0.3
```

---

### Text Summarization

```
--text_path
--max_sentences
--tfidf_max_features
--max_comp
```

Example:

```bash
python src/experiments/run.py \
    --suite realworld \
    --problem textsum \
    --text_path data/article.txt
```

---

## Multiple Seeds

Comparison experiments support running multiple seeds:

```
--seeds
```

Example:

```
--seeds 42,43,44,45,46
```

---

## Saving Results

Results can be saved using:

```
--save
```

Optional parameters:

```
--out
--tag
```

Example:

```bash
python src/experiments/run.py \
    --suite benchmark \
    --problem zdt1 \
    --save
```

Output will be stored in:

```
experiments/results/
```

Saved artifacts include:

- Pareto fronts
- convergence curves
- metrics
- configuration files
- solution sets

---

## Visualization

The framework automatically generates plots such as:

- Pareto front (2D and 3D)
- convergence curves
- feature selection accuracy vs number of features
- project profit vs risk tradeoff
- text summarization objective plots

Plots are either:

- displayed interactively
- saved to disk when `--save` is enabled.
---

# Documentation

Detailed documentation of the project, including:

- theoretical background of the MOEA/D algorithm
- description of implemented benchmark problems
- experimental setup and parameter analysis
- analysis of obtained results

is available in the **project report written in Serbian**, located in the `docs/` directory.

# Authors

Jovana Brkljac  
Mateja Janic
