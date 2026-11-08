# Iris Classification Report

## Overview

This experiment implements a neural network from scratch in C++ to classify Iris flowers using four numerical measurements. The implementation is in [`ImageClassification.cpp`](../src/MachineLearning/ImageClassification.cpp).

The dataset contains 150 samples across three species: setosa, versicolor, and virginica. The data and accompanying description are available in [`dataset/iris`](../dataset/iris), including the original [`iris.names`](../dataset/iris/iris.names) documentation.

## Configuration

- Architecture: 4 input features, 32 hidden neurons with ReLU, and 3 output classes with softmax.
- Training: backpropagation with mini-batch gradient descent and cross-entropy loss.
- Learning rate: 0.05; epochs: 500; batch size: 32.
- Random seed: 42 for weight initialization and batch shuffling.
- Inputs: raw feature values without normalization.

## Results After Optimization

The optimized model achieved **98% training accuracy**, correctly classifying **147 of 150 samples**. Running the current implementation reproduced this result, with a final training cross-entropy loss of approximately **0.0440**.

All 150 samples were used for both training and evaluation. This result measures performance on the training set; accuracy on unseen data has not yet been evaluated. A next step is to introduce a separate test set to measure generalization.

## Reproduce

From the repository root, using a GCC C++ compiler with C++17 support:

```powershell
g++ -std=c++17 -O2 src/MachineLearning/ImageClassification.cpp -o src/MachineLearning/ImageClassification.exe
.\src\MachineLearning\ImageClassification.exe
```

The program locates `dataset/iris/iris.data` automatically when run inside the project. An alternative dataset path can be passed as the first command-line argument.
