# Iris Classification Report

## Overview

This report describes the earlier Iris experiment. Its [implementation at commit e3416b1](https://github.com/phutrong-2208/AI_From_Scratch/blob/e3416b17a8f400a58fa0419773fd71f093cede6e/src/MachineLearning/ImageClassification.cpp) classifies Iris flowers using four numerical measurements. The current [SGD and momentum versions](letter-recognition.md) use the Letter Recognition dataset and are undergoing optimization; the 98% training accuracy below belongs to the Iris experiment.

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
