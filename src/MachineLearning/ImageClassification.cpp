#include "bits/stdc++.h"
#include <filesystem>

using namespace std;

///Hyperparameters
const double lr = 0.05;
const int INPUT_SIZE = 4;
const int HIDDEN_SIZE = 8;
const int OUTPUT_SIZE = 1;
const int EPOCHS = 500;
const int BATCH_SIZE = 32;
const int SEED = 1;


///Definitions
using Vector = vector<double>;
using Matrix = vector<Vector>;


/// Dataset Loader
struct Dataset{
    Matrix X;
    vector<int> y;
};


int labelIndex(const string &species){
    if(species == "Iris-setosa") return 0;
    if(species == "Iris-versicolor") return 1;
    if(species == "Iris-virginica") return 2;
    throw runtime_error("Unknown species: " + species);
}

filesystem::path defaultDatasetPath() {
    // Relative paths are resolved from the working directory, not this source file.
    filesystem::path directory = filesystem::current_path();
    while (true) {
        filesystem::path candidate = directory / "dataset" / "iris" / "iris.data";
        if (filesystem::is_regular_file(candidate)) return candidate;
        filesystem::path parent = directory.parent_path();
        if (parent == directory) break;
        directory = parent;
    }
    throw runtime_error("Cannot find dataset/iris/iris.data from working directory: "
                        + filesystem::current_path().string()
                        + ". Run inside the project or pass the dataset path as an argument.");
}

Dataset loadDataset(const string& filename) {
    ifstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filesystem::absolute(filename).string());
    }

    Dataset dataset;
    string line;

    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        string value;
        Vector features;

        // Read 4 numerical features
        for (int i = 0; i < 4; ++i) {
            if (!getline(ss, value, ',')) {
                throw runtime_error("Invalid data row");
            }

            features.push_back(stod(value));
        }

        // Read species name
        string species;
        getline(ss, species);

        if (!species.empty() && species.back() == '\r') {
            species.pop_back();
        }

        dataset.X.push_back(features);
        dataset.y.push_back(labelIndex(species));
    }

    if (dataset.X.empty()) {
        throw runtime_error("Dataset is empty: " + filename);
    }
    return dataset;
}

///Matrix Operations
Matrix operator * (const Matrix &A, const Matrix &B){
    if(A[0].size() != B.size()) throw invalid_argument("Invalid multiplication");
    Matrix F(A.size(), vector<double>(B[0].size()));
    for (int i = 0; i < A.size(); ++i){
        for (int j = 0; j < B[0].size(); ++j){
            for (int k = 0; k < A[0].size(); ++k){
                F[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return F;
}

Matrix operator - (const Matrix &A, const Matrix &B){
    if(A.size() != B.size() || A[0].size() != B[0].size()) throw invalid_argument("Invalid subtraction");

    Matrix F(A.size(), Vector(A[0].size()));

    for (int i = 0; i < A.size(); ++i){
        for (int j = 0; j < A[0].size(); ++j){
            F[i][j] = A[i][j] - B[i][j];
        }
    }
    return F;
}
Matrix operator * (const Matrix& A, double m){
    Matrix F(A.size(), Vector(A[0].size()));
    for (int i = 0; i < A.size(); ++i){
        for (int j = 0; j < A[0].size(); ++j){
            F[i][j] = A[i][j] * m;
        }
    }
    return F;
}

Matrix tranpose(const Matrix& A){
    Matrix result(A[0].size(), Vector(A.size()));

    for (int i = 0; i < A[0].size(); ++i){
        for (int j = 0; j < A.size(); ++j){
            result[i][j] = A[j][i];
        }
    }
    return result;
}

Vector sumRow(const Matrix& A){
    Vector result;
    for (int j = 0; j < A[0].size(); ++j){
        double sum = 0.0;
        for (int i = 0; i < A.size(); ++i){
            sum += A[i][j];
        }
        result.emplace_back(sum);
    }
    return result;
}

///Vector Operations

Vector operator - (const Vector& A, const Vector& B){
    if(A.size() != B.size()){
        throw invalid_argument("Invalid vector subtraction");
    }
    Vector result(A.size());
    for (int i = 0; i < A.size(); ++i){
        result[i] = A[i] - B[i];
    }
    return result;
}

Vector operator * (const Vector& A, double coeff){
    Vector result(A.size());
    for (int i = 0; i < A.size(); ++i){
        result[i] = A[i] * coeff;
    }
    return result;
}



///Neural Network


class NeuralNetwork{
private:
    int inputSize, hiddenSize, outputSize;

    Matrix W1;
    Matrix W2;

    Vector b1;
    Vector b2;

    Matrix Z1, A1, y;
public:
    NeuralNetwork(int input, int hidden, int output): inputSize(input), hiddenSize(hidden), outputSize(output),
        W1(input, vector<double> (hidden)), W2(hidden, vector<double> (output)),
        b1(hidden, 0.0f), b2(output, 0.0f){

        mt19937 rng(42);
        normal_distribution<double> dist(0.0f, 2.0f);


        for (int i = 0; i < inputSize; ++i){
            for (int j = 0; j < hiddenSize; ++j){
                W1[i][j] = dist(rng);
            }
        }
        for (int i = 0; i < hiddenSize; ++i){
            for (int j = 0; j < outputSize; ++j){
                W2[i][j] = dist(rng);
            }
        }
    }


    Matrix addBias(const Matrix &A, const Vector &B){
        Matrix result = A;
        for (int i = 0; i < A.size(); ++i){
            if(A[i].size() != B.size()) throw invalid_argument("Bias dimension mismatch");
            for (int j = 0; j < A[0].size(); ++j){
                result[i][j] += B[j];
            }
        }
        return result;
    }

    Matrix ReLU(const Matrix& A){
        Matrix result = A;

        for (int i = 0; i < A.size(); ++i){
            for (int j = 0; j < A[0].size(); ++j){
                result[i][j] = max(0.0, result[i][j]);
            }
        }
        return result;
    }

    Matrix Softmax(const Matrix& A){
        Matrix result = A;

        for (int i = 0; i < A.size(); ++i){
            double maxLogit = *max_element(A[i].begin(), A[i].end());

            double sumExp = 0.0f;
            for (int j = 0; j < A[0].size(); ++j){
                sumExp += exp(A[i][j] - maxLogit);
            }
            for (int j = 0; j < A[0].size(); ++j){
                result[i][j] = (double)exp(A[i][j] - maxLogit) / sumExp;
            }
        }
        return result;
    } // This function to transform a value into probability


    Matrix forward(const Matrix& X){
        Z1 = addBias(X * W1, b1); //--> (2, 4) x (4, 8) (2, 8)
        A1 = ReLU(Z1);

        Matrix Z2 = addBias(A1 * W2, b2);

        return Softmax(Z2);
    }

    double crossEntropy(const Matrix& y_bar, const vector<int>& label){
        if(y_bar.size() == 0 || label.size() != y_bar.size()) {
            throw invalid_argument("Invalid calculation for loss function ");
        }

        double loss = 0.0f;
        int n_samples = (int)y_bar.size();


        for (int i = 0; i < n_samples; ++i){
            int correctClass = label[i];
            if(correctClass < 0 || correctClass >= y_bar.size()) throw invalid_argument("Wrong initial label");

            double p = y_bar[i][correctClass];
            p = max(p, 1e-12); //Prevent log(0) case

            loss -= log(p);
        }

        return loss / n_samples;
    }

    void backward(const Matrix& X, const vector<int>& Y, double lr){
        int m = X.size();

        if (m == 0 || m != Y.size())
            throw invalid_argument("Invalid batch");

        Matrix y_bar = forward(X);

        Matrix y(y_bar.size(), Vector(y_bar[0].size(), 0.0));

        for (int i = 0; i < Y.size(); ++i) y[i][Y[i]] = 1;

        //Output Layer
        Matrix dL_Z2 = (y_bar - y) * (1.0 / m);

        //Gradient of W2 and b2
        Matrix dL_W2 = tranpose(A1) * dL_Z2;
        Vector dL_b2 = sumRow(dL_Z2);

        //Back propagation to hidden

        Matrix dL_A1 = dL_Z2 * tranpose(W2);
        Matrix dL_Z1 = dL_A1;

        for (int i = 0; i < m; ++i){
            for (int j = 0; j < hiddenSize; ++j){
                if(Z1[i][j] <= 0.0) dL_Z1[i][j] = 0.0;
            }
        } //Partial Derivative of ReLU


        //Gradient of W1 and b1
        Matrix dL_W1 = tranpose(X) * dL_Z1;
        Vector dL_b1 = sumRow(dL_Z1);


        //Update new weights and biases
        W1 = W1 - dL_W1 * lr;
        b1 = b1 - dL_b1 * lr;

        W2 = W2 - dL_W2 * lr;
        b2 = b2 - dL_b2 * lr;
    }

    void train(const Matrix& X, const vector<int> &Y){
        int n_samples = X.size();
        vector<int> ord(n_samples);
        iota(ord.begin(), ord.end(), 0);

        mt19937 rng(42);

        for (int epoch = 0; epoch < EPOCHS; ++epoch){
            shuffle(ord.begin(), ord.end(), rng);

            for (int batch = 0; batch <= n_samples; batch += BATCH_SIZE){
                int l = batch;
                int r = min(batch + BATCH_SIZE, n_samples);

                Matrix batchX;
                vector<int> batchY;
                for (int k = l; k < r; ++k){
                    int index = ord[k];

                    batchX.push_back(X[index]);
                    batchY.push_back(Y[index]);
                }
                backward(batchX, batchY, lr);
            }


            if((epoch + 1) % 50 == 0 || epoch == 0){
                Matrix prediction = forward(X);

                double loss = crossEntropy(prediction, Y);

                cout << "Epoch " << epoch + 1 << "/" << EPOCHS << " | Loss: " << loss << '\n';
            }
        }
    }
};



double accuracy(const Matrix& X, const vector<int>& label, NeuralNetwork &model){
    if(X.empty() || X.size() != label.size()) {
        throw invalid_argument("Invalid evaluation data");
    }
    Matrix predict = model.forward(X);
    int correctness = 0;

    for (int i = 0; i < (int)predict.size(); ++i){
        int predictMax = (int)(max_element(predict[i].begin(), predict[i].end()) - predict[i].begin());

        if(predictMax == label[i]){
            correctness++;
        }
    }
    return (double)correctness / (double)X.size();
}

int main(int argc, char* argv[]){
    try{
        filesystem :: path datasetPath = argc > 1 ? filesystem::path(argv[1]) : defaultDatasetPath();
        Dataset data = loadDataset(datasetPath.string());

        cout << "Sample: " << data.X.size() << '\n';
        cout << "Features: " << data.X[0].size() << '\n';

        cout << "First sample: [";
        for (size_t i = 0; i < data.X[0].size(); ++i) {
            if (i > 0) cout << ", ";
            cout << data.X[0][i];
        }
        cout << "]\n";

        NeuralNetwork model(4, 32, 3);


        model.train(data.X, data.y);

        cout << "Accuracy of Training Data: " << accuracy(data.X, data.y, model) * 100 << "% \n";
    }
    catch (const exception& e){
        cerr << e.what() << '\n';
        return 1;
    }

    return 0;
}

