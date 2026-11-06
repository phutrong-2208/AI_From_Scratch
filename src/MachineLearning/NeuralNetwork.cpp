#include "bits/stdc++.h"

using namespace std;

using Vector = vector<double>;
using Matrix = vector<Vector>;

class NeuralNetwork{
private:
    int inputSize, hiddenSize;
    bool classification;

    Matrix W1;
    Vector b1, W2, h;
    double b2 = 0.0f;

    double sigmoid(double x){
        return 1.0 / (1 + exp(-x));
    }

public:
    NeuralNetwork(int inputSize, int hiddenSize, bool isClassification) :
        inputSize(inputSize), hiddenSize(hiddenSize), classification(isClassification),
        W1(hiddenSize, vector<double>(inputSize)), b1(hiddenSize, 0.0f), W2(hiddenSize), h(hiddenSize){
            mt19937 rng(42);

            uniform_real_distribution<double> dist(-1.0f, 1.0f);

            for (int i = 0; i < hiddenSize; ++i){
                for (int j = 0; j < inputSize; ++j){
                    W1[i][j] = dist(rng);
                }
                W2[i] = dist(rng);
            }
        }

    double forward(const Vector& x){
        //From input to hidden layers
        for (int i = 0; i < hiddenSize; ++i){
            double z = b1[i];
            for (int j = 0; j < inputSize; ++j){
                z += W1[i][j] * x[j];
            }
            h[i] = sigmoid(z);
        }


        //From hidden layers to output

        double z2 = b2;

        for (int i = 0; i < hiddenSize; ++i){
            z2 += W2[i] * h[i];
        }
        return classification ? sigmoid(z2) : z2;
    }

    void backward(const Vector& x, double y, double lr){
        double y_hat = forward(x);

        double delta2 = y_hat - y;
        if(classification){
            delta2 *= y_hat * (1 - y_hat);
        }

        vector<double> delta1(hiddenSize);
        for (int i = 0; i < hiddenSize; ++i){
            delta1[i] = delta2 * h[i] * (1 - h[i]) * W2[i];
        }

        for (int i = 0; i < hiddenSize; ++i){
            W2[i] -= h[i] * delta2 * lr;
        }

        b2 -= delta2 * lr;

        for (int i = 0; i < hiddenSize; ++i){
            for (int j = 0; j < inputSize; ++j){
                W1[i][j] -= delta1[i] * x[j] * lr;
            }
            b1[i] -= lr * delta1[i];
        }
    }
    void train(const Matrix& X, const Vector& Y, int epochs, double lr){
        for (int epoch = 0; epoch < epochs; ++epoch){
            for (int i = 0; i < (int)X.size(); ++i){
                backward(X[i], Y[i], lr);
            }
        }
    }
};

int main(void){
    Matrix X = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    Vector Y = {
        0,
        1,
        1,
        0
    };

    NeuralNetwork nn(2, 4, true);

    nn.train(X, Y, 500000, 0.05);

    cout << fixed << setprecision(5);

    for (int i = 0; i < 4; ++i){
        cout << X[i][0] << " XOR " << X[i][1] << " = " << nn.forward(X[i]) << " (target " << Y[i] << ")\n";
    }
}
