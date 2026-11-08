# AI From Scratch

My personal notebook documenting my learning journey in **Machine Learning**, **Deep Learning**, and **Computer Vision**. It contains notes, mathematical foundations, and algorithms I implement from scratch to understand how they work.

## Current Contents

- [`Notebook.tex`](Notebook.tex): theoretical notes written in LaTeX.
- [`StochasticGradientDescent.cpp`](src/MachineLearning/StochasticGradientDescent.cpp): linear regression using Stochastic Gradient Descent.
- [`NeuralNetwork.cpp`](src/MachineLearning/NeuralNetwork.cpp): a neural network with one hidden layer, trained using backpropagation on the XOR problem.
- [`ImageClassification.cpp`](src/MachineLearning/ImageClassification.cpp): a neural network implemented from scratch, now using the **Letter Recognition** dataset with 16 features and 26 classes (A-Z). The current experiment uses 16,000 training samples and 4,000 test samples. **Optimization is in progress.**

The earlier Iris experiment achieved **98% training accuracy**; its configuration and results are recorded in the [Iris report](reports/iris-classification.md).

I will keep adding notes and implementations as I learn. This repository contains source code and notes; PDF files and build artifacts are excluded.
