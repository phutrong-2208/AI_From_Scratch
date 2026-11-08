# AI From Scratch

My personal notebook documenting my learning journey in **Machine Learning**, **Deep Learning**, and **Computer Vision**. It contains notes, mathematical foundations, and algorithms I implement from scratch to understand how they work.

## Current Contents

- [`Notebook.tex`](Notebook.tex): theoretical notes written in LaTeX.
- [`StochasticGradientDescent.cpp`](src/MachineLearning/StochasticGradientDescent.cpp): linear regression using Stochastic Gradient Descent.
- [`NeuralNetwork.cpp`](src/MachineLearning/NeuralNetwork.cpp): a neural network with one hidden layer, trained using backpropagation on the XOR problem.
- [`ImageClassification_version_1.cpp`](src/MachineLearning/ImageClassification_version_1.cpp): Letter Recognition using **SGD**.
- [`ImageClassification_version_2.cpp`](src/MachineLearning/ImageClassification_version_2.cpp): Letter Recognition using **SGD + Momentum**.

The Letter Recognition experiments use 16 features, 26 classes (A-Z), and a 16,000/4,000 training/test split. The best mean test accuracy among the tested configurations was **82.55 ± 2.53%** with SGD at a learning rate of 0.01 across three seeds. See the [experiment report](reports/letter-recognition.md) for the initialization comparison and optimizer results. **Optimization is in progress.**

The earlier Iris experiment achieved **98% training accuracy**; its configuration and results are recorded in the [Iris report](reports/iris-classification.md).

I will keep adding notes and implementations as I learn. This repository contains source code and notes; PDF files and build artifacts are excluded.
