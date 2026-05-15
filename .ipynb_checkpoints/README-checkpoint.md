![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Neural Networks Fundamentals

## Overview

In today's lesson you saw what an artificial neuron is, why activation functions matter, and how forward propagation turns inputs into predictions. In this lab you'll make all of that real by building a small neural network in two ways:

1. **From scratch** in NumPy — so the matrix multiplications, activations, and shapes are tangible, not magical.
2. **In PyTorch** — so you finish the day comfortable with the framework you'll use for the rest of the unit.

You'll then run small experiments to feel how the network's choices (number of hidden units, choice of activation) change its behaviour on a real classification dataset.

## Learning Goals

By the end of this lab you should be able to:

- Implement forward propagation for an MLP using only NumPy operations.
- Translate the same architecture into PyTorch using `nn.Module` and `nn.Linear`.
- Verify that a NumPy and a PyTorch network with the **same weights** produce identical outputs.
- Reason about how the number of hidden units and the choice of activation function affect a model's predictions.

## Setup and Context

You'll work in a single Jupyter Notebook. The dataset is the **Breast Cancer Wisconsin** dataset bundled with scikit-learn — 30 numeric features, binary target (malignant / benign). It's small enough that a tiny MLP can do interesting work.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install numpy pandas matplotlib scikit-learn torch
```

If the PyTorch install fails on your platform, follow the official guide at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) and pick the CPU build.

## Getting Started

1. Create a notebook called **`m6-01-neural-networks-fundamentals.ipynb`** in the repo root.
2. Start every notebook session with the same imports:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
torch.manual_seed(42)
```

3. Work through the tasks in order. Add a markdown cell after each task with a short interpretation of what you observed.

## Tasks

### Task 1 — A Single Neuron in NumPy

You'll start with one neuron — a logistic regression in disguise.

1. Load the Breast Cancer dataset, split into train/test (`test_size=0.2`, `random_state=42`), and scale the features with `StandardScaler`.
2. Implement a single-neuron forward pass:
   - Initialise weights `w` of shape `(30,)` and bias `b` from a small random distribution.
   - Implement `forward(x, w, b)` that returns `sigmoid(w·x + b)`.
   - Run it on the first 5 test rows. Report the predicted probabilities.
3. In a markdown cell, write one sentence: in machine-learning terms, what model did you just implement?

### Task 2 — A Two-Layer MLP in NumPy

Now stack neurons into a real network.

1. Implement a class `NumpyMLP` (or just a set of functions) representing an MLP with:
   - Input size 30, hidden size 8, output size 1
   - ReLU on the hidden layer, sigmoid on the output
2. Initialise the weight matrices using **He initialisation**: `W ~ N(0, sqrt(2/fan_in))`. Initialise biases to zero.
3. Implement a `forward(X)` that takes a batch `(N, 30)` and returns predictions `(N, 1)`. Use vectorised NumPy — no Python loops over rows.
4. Run a forward pass on the test set. Print the shape of the output and the first 5 predictions.

**Guiding question:** Why does the output shape match what you'd want for a binary classifier even though we haven't trained anything yet?

### Task 3 — The Same Network in PyTorch

Recreate Task 2 using PyTorch's high-level API.

1. Define a class `TorchMLP(nn.Module)` with the same architecture (30 → 8 → 1, ReLU then Sigmoid).
2. Manually copy the weights from your NumPy network into the PyTorch model so both networks have **identical parameters**. Hint: `model.fc1.weight.data = torch.from_numpy(W1).float()` and similar for biases. Be careful with shapes — `nn.Linear` stores weights as `(out, in)`.
3. Run the same test batch through the PyTorch model. Compare the outputs to the NumPy outputs. They should match to at least 6 decimal places.
4. Report the maximum absolute difference between the two outputs.

**Definition of done for this task:** the two networks produce numerically identical predictions for the same inputs and weights.

### Task 4 — Activation Function Experiment

Time to experiment.

1. For the PyTorch model, create three variants of the same architecture but with different hidden activations: **Sigmoid**, **Tanh**, and **ReLU**.
2. For each variant, run a forward pass on the **same** scaled test set.
3. Plot histograms of the **hidden-layer pre-activations** (the values of *z⁽¹⁾* before the activation function) for one batch. Use a 1×3 subplot.
4. Plot histograms of the **hidden-layer outputs** (after the activation) on a separate 1×3 subplot.
5. In a markdown cell, answer:
   - For Sigmoid and Tanh, what fraction of the post-activation values fall in the saturated region (close to 0/1 for sigmoid, close to ±1 for tanh)?
   - Looking at ReLU: roughly what fraction of the units are inactive (output exactly 0)?
   - Based on these histograms, why might ReLU be the better default for hidden layers?

## Submission

### What to submit

- `m6-01-neural-networks-fundamentals.ipynb` — completed notebook, runs top-to-bottom without errors.

### Definition of done (checklist)

- [ ] Single-neuron forward pass implemented and run on the test set.
- [ ] NumPy MLP (30 → 8 → 1) with He initialisation runs correctly on a batch.
- [ ] PyTorch MLP with manually copied weights produces outputs identical to the NumPy MLP.
- [ ] Activation experiment shows histograms for sigmoid, tanh, and ReLU and includes a written interpretation.
- [ ] All tasks have a brief markdown reflection.
- [ ] `Kernel → Restart & Run All` produces no errors.

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete neural networks fundamentals"
git push origin main
```

Then open a **Pull Request** on the original repository with a short description of what you built.
