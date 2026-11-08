# Letter Recognition — Experiment Report

## Dataset and Setup

This project implements a neural network from scratch in C++ for the Letter Recognition dataset. It contains **20,000 samples**, each represented by **16 numerical features** and a capital-letter label from **A to Z**. The inputs are extracted image features, rather than raw pixel images.

The first **16,000 samples** are used for training and the remaining **4,000** for testing. The model has **16 inputs, 32 hidden neurons with ReLU, and 26 softmax outputs**, trained with cross-entropy loss for **100 epochs** and a **batch size of 128**. Both versions currently use the original feature values in `[0, 15]`.

- [Version 1 — SGD](../src/MachineLearning/ImageClassification_version_1.cpp): mini-batch stochastic gradient descent.
- [Version 2 — SGD + Momentum](../src/MachineLearning/ImageClassification_version_2.cpp): adds velocity updates with a momentum coefficient of **0.9**.

The results below were recorded during my experiments; they were not recomputed for this report.

## Experiment 1 — He Initialization

Initially, I used Gaussian weight initialization with a standard deviation of **2.0**, resulting in approximately **16% test accuracy**. After switching to He initialization, a subsequent run achieved **82.03% test accuracy**. The current implementations initialize weights with a standard deviation of `sqrt(2 / fan_in)`.

This helped me understand the importance of choosing an appropriate weight initialization strategy. The comparison was exploratory rather than a fully controlled experiment, so the improvement cannot be attributed to initialization alone with certainty.

## Experiment 2 — Optimizer and Learning Rate

I compared learning rates and optimizers across **three random seeds**. Results are reported as mean test accuracy ± standard deviation.

| Optimizer | Learning Rate | Test Accuracy (Mean ± Std) |
| --- | ---: | ---: |
| SGD | 0.1 | 77.73 ± 3.44% |
| **SGD** | **0.01** | **82.55 ± 2.53%** |
| SGD + Momentum | 0.01 | 69.38 ± 3.93% |
| SGD + Momentum | 0.001 | 78.49 ± 1.18% |
| SGD + Momentum | 0.003 | 79.94 ± 2.70% |

**SGD at a learning rate of 0.01 achieved the highest mean test accuracy among the tested configurations.** Momentum performed better at the smaller learning rates tested, but did not exceed that SGD result. Momentum at 0.001 had the lowest observed standard deviation across the three seeds.

These runs suggest that learning-rate choice matters for both optimizers and that adding momentum does not automatically improve accuracy. With only three seeds, these observations remain preliminary.

## Reproducibility and Status

The source files are experiment snapshots, rather than a script that runs every configuration in the table. Version 1 currently uses a learning rate of **0.1** and a fixed initialization seed of **42**. Version 2 uses a learning rate of **0.003** and initialization seeds **1–3**. Both use a fixed batch-shuffling seed of **42**; the momentum version computes sample standard deviation with denominator `n - 1`.

Reproducing the full table requires setting the learning rate and initialization seeds for each configuration. The 82.03% result in Experiment 1 is a single-run observation, while Experiment 2 reports averages across three seeds. Since learning rates were compared using test accuracy, the test set also informed configuration selection.

**Optimization is still in progress.** These experiments document my learning process and the results observed so far.
