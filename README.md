# SwiftFL

This repository contains the official artifact for:

**SwiftFL: Enabling Speculative Training for On-Device Federated Deep Learning**

SwiftFL is implemented on top of FederatedScope and enables speculative training with gradient prediction and compensation for federated learning systems.

---

# Quick Start

## 1. Installation

### Install Miniconda (if not already installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py39_23.1.0-1-Linux-x86_64.sh
source ~/.bashrc
```

---

### Create a clean environment

We recommend creating a dedicated conda environment:

```bash
conda create -n swiftfl python=3.9
conda activate swiftfl
```

---

### Install PyTorch (CUDA 11.3 version)

```bash
conda install -y pytorch=1.10.1 torchvision=0.11.2 \
torchaudio=0.10.1 torchtext=0.11.1 cudatoolkit=11.3
```

If you do not have GPU support, install the CPU-only version of PyTorch.

---

### Install SwiftFL

```bash
cd SwiftFL
pip install -e .[dev,app]
```

This will install SwiftFL and its dependencies.

---

# Experiment E1: Minimal End-to-End Example

## Purpose

The original large-scale experiments in the paper involve thousands of simulated clients and require up to 214 hours for full reproduction.

To facilitate artifact evaluation, we provide a **minimal reproducible example**:

* Model: ConvNet2
* Dataset: FEMNIST
* Total number of clients reduced by 10×
* All other hyperparameters remain unchanged

This configuration preserves heterogeneity and speculative execution behavior while completing in minutes.

---

## Run the Example

Configuration files are located under:

```
scripts/example/
```

Run:

```bash
python federatedscope/main.py \
  --cfg scripts/example/swiftfl_convnet2_on_femnist.yaml
```

---

## Expected Output

Results are stored under:

```
exp/femnist/...
```

Key log files:

* `eval_results.log`

  * Per-round training time
  * Loss
  * Accuracy
    (Used to reproduce Figure 8 trends)

* `system_metrics.log`

  * Per-client training time
  * Per-client accuracy

* `waiting_rate.log`

  * Per-client waiting time
  * Per-client computation time
    (Used to reproduce Figure 10 trends)

* `eval_results.raw.gz`

  * Fine-grained per-client and per-step statistics
  * Participating clients per round
  * Step-level loss and timing

Successful execution should show:

* Reduced waiting time for fast clients
* Stable convergence
* Comparable final accuracy to standard FL

---

# Experiment E2: Comparison with Other FL Methods

This experiment corresponds to **Section 7.2, Table 3 and Figure 9** in the paper.

All configurations are located under:

```
scripts/baselines/
```

These include SwiftFL and the following baseline systems:

* FjORD
* FLuID
* Papaya
* PyramidFL

---

## Run a Specific Configuration

Example: SwiftFL with ConvNet2 on FEMNIST

```bash
python federatedscope/main.py \
  --cfg scripts/baselines/femnist/swiftfl_convnet2_on_femnist.yaml
```

---

## Batch Execution

Each dataset directory contains a `run.sh` script that executes all configurations sequentially:

```
scripts/baselines/<dataset>/run.sh
```

Full reproduction requires multi-GPU hardware and takes approximately:

* **88 hours** on a 2×A100 server

Running a subset of models or datasets is sufficient to observe the performance trends reported in the paper.

---

# Experiment E3: Evaluation Across FDL Optimizers

This experiment corresponds to **Section 7.2, Table 2**.

Configurations are located under:

```
scripts/optimizers/
```

Two subdirectories are provided:

* `w_swiftfl/` — Optimizers with SwiftFL enabled
* `w_o_swiftfl/` — Optimizers without SwiftFL

Supported optimizers include:

* FedAdam
* FedNova
* FedYogi
* Scaffold

---

## Execution

Run any configuration as:

```bash
python federatedscope/main.py --cfg <path_to_yaml>
```

Full reproduction takes approximately:

* **164 hours** on recommended multi-GPU hardware

---

## Expected Output

Output format is identical to E1.

To validate optimizer generality:

* Compare convergence time
* Compare final accuracy
* Observe relative speedup trends

Running a subset of optimizers is sufficient to reproduce the behavior described in the paper.

---

# Hardware Requirements

* Minimal example (E1):

  * Single GPU (recommended) or CPU-only
  * 64GB RAM
* Full reproduction (E2 + E3):

  * 2×A100 GPUs
  * 1TB RAM
  * 512GB disk

