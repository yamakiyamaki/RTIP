#include <iostream>
#include <vector>

using namespace std;

using Matrix = vector<vector<double>>;

// Function to compute mean of each column
vector<double> computeMean(const Matrix& data) {
    int n = data.size();       // number of data points
    int d = data[0].size();    // dimension
    vector<double> mean(d, 0.0);

    for (const auto& row : data) {
        for (int i = 0; i < d; ++i) {
            mean[i] += row[i];
        }
    }
    for (int i = 0; i < d; ++i) {
        mean[i] /= n;
    }
    return mean;
}

// Function to compute covariance matrix
Matrix computeCovarianceMatrix(const Matrix& data) {
    int n = data.size();       // number of samples
    int d = data[0].size();    // number of dimensions (e.g., 3 for RGB)
    Matrix cov(d, vector<double>(d, 0.0));

    vector<double> mean = computeMean(data);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            for (int k = 0; k < d; ++k) {
                cov[j][k] += (data[i][j] - mean[j]) * (data[i][k] - mean[k]);
            }
        }
    }

    for (int j = 0; j < d; ++j)
        for (int k = 0; k < d; ++k)
            cov[j][k] /= (n - 1); // unbiased estimate

    return cov;
}

// Helper to print a matrix
void printMatrix(const Matrix& mat) {
    for (const auto& row : mat) {
        for (double val : row) {
            cout << val << "\t";
        }
        cout << endl;
    }
}

// Example usage
int main() {
    // Each row is a data point: [R, G, B]
    Matrix data = {
        {6, 1, 1},
        {4, -2, 5},
        {2, 8, 7},
        {5, 3, 2},
        {7, 1, 6}
    };

    Matrix covMatrix = computeCovarianceMatrix(data);

    cout << "Covariance Matrix:" << endl;
    printMatrix(covMatrix);

    return 0;
}