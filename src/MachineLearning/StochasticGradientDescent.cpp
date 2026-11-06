#include "bits/stdc++.h"

using namespace std;

mt19937 rng(chrono :: steady_clock :: now().time_since_epoch().count());

int main(void){
    int N = 5;
    vector<int> x = {1, 2, 3, 4, 5};
    vector<int> y = {2, 3, 7, 9, 11};


    double the_ta0 = 0;
    double the_ta1 = 0;

    int epoch = 1000;

    double alpha = 0.01f;

    vector<int> ord(N);
    iota(ord.begin(), ord.end(), 0);

    /// We have the linear regression: y = h(x) = the_ta0 + the_ta1 * x
    /// Error: e = 1/2 (y^ - y)^2

    for (int i = 0; i < epoch; ++i){
        shuffle(ord.begin(), ord.end(), rng);

        for (auto j : ord){
            double y_prime = the_ta0 + the_ta1 * x[j];
            double e = (y_prime - y[j]);

            the_ta0 -= 1.0f * alpha * e;
            the_ta1 -= 1.0f * alpha * e * x[j];
        }
        cout << the_ta0 << " " << the_ta1 << '\n';
    }

}


