#include <iostream>
#include <vector>
#include <algorithm>  // for std::next_permutation

using namespace std;

// Calculate the sign (+1 or -1) of a permutation using inversion count
int permutationSign(const vector<int>& perm) {
    int sign = 1;
    int n = perm.size();
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (perm[i] > perm[j]) {
                sign *= -1;
            }
        }
    }
    return sign;
}

// Calculate determinant using Leibniz formula
double determinant(const vector<vector<double>>& A) {
    int n = A.size();
    vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;

    double det = 0.0;

    do {
        double term = 1.0;
        for (int i = 0; i < n; ++i) {
            term *= A[i][perm[i]];
        }
        det += permutationSign(perm) * term;
    } while (next_permutation(perm.begin(), perm.end()));

    return det;
}

int main() {
    vector<vector<double>> A = {
        {6, 1, 1},
        {4, -2, 5},
        {2, 8, 7}
    };

    cout << "Determinant (Leibniz formula): " << determinant(A) << endl;
    return 0;
}