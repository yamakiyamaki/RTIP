#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>  // for high_resolution_clock
#include <vector>
#include <algorithm>  // for std::next_permutation


using namespace std;
using Matrix = vector<vector<double>>;

// // Calculate the sign (+1 or -1) of a permutation using inversion count
// int permutationSign(const vector<int>& perm) {
//   int sign = 1;
//   int n = perm.size();
//   for (int i = 0; i < n; ++i) {
//       for (int j = i + 1; j < n; ++j) {
//           if (perm[i] > perm[j]) {
//               sign *= -1;
//           }
//       }
//   }
//   return sign;
// }

// // Calculate determinant using Leibniz formula (https://ja.wikipedia.org/wiki/%E8%A1%8C%E5%88%97%E5%BC%8F)
// double determinant(const vector<vector<double>>& A) {
//   int n = A.size();
//   vector<int> perm(n);
//   for (int i = 0; i < n; ++i) perm[i] = i;

//   double det = 0.0;

//   do {
//       double term = 1.0;
//       for (int i = 0; i < n; ++i) {
//           term *= A[i][perm[i]];
//       }
//       det += permutationSign(perm) * term;
//   } while (next_permutation(perm.begin(), perm.end()));

//   return det;
// }

double determinant(const vector<vector<double>>& A) {
  if (A.size() == 3 && A[0].size() == 3) {
      return
          A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
          A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
          A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  } else {
      // cerr << "Only 3x3 matrices supported for determinant." << endl;
      return 0.0;
  }
}

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
Matrix computeCovarianceMatrix(const Matrix& data, Matrix& cov, int n, int d) {
    // int n = data.size();       // number of samples
    // int d = data[0].size();    // number of dimensions (e.g., 3 for RGB)
    // Matrix cov(d, vector<double>(d, 0.0));

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

    // return cov;
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

float calc_kSize(const cv::Mat_<cv::Vec3b>& source, int i, int j, float neighborSize, float factor) {
  // Extract neighborhood
  int nSize = static_cast<int>(neighborSize * neighborSize);
  Matrix neighbor(nSize, std::vector<double>(3));
  // float factor = 10.0f;
  int neighborDiv2 = static_cast<int>(neighborSize / 2.0);
  int n = 0;
  for (int dy = -neighborDiv2; dy <= neighborDiv2; ++dy) {
    for (int dx = -neighborDiv2; dx <= neighborDiv2; ++dx) {
      int y = i + dy;
      int x = j + dx;
      // Bounds check
      if (y < 0 || y >= source.rows || x < 0 || x >= source.cols) {
        continue;  // or handle with zero padding etc.
      }
      // if (n >= neighbor.size()) {
      //   // cerr << "Warning: trying to access out-of-bounds neighbor[" << n << "]" << endl;
      //   continue;
      // }

      // for (int c = 0; c < 3; ++c){
      //   neighbor[n][c] = source(y, x)[c];
      // }
      neighbor[n] = source(y, x);
      n++;
    }
  }
  // Compute covariance matrix
  // Matrix covMatrix = computeCovarianceMatrix(neighbor);
  // printMatrix(covMatrix);

  // Compute determinant
  // float determ = determinant(covMatrix);
  if (n == 0) {
    return 3.0f;  // Fallback: no valid neighbors
  }
  neighbor.resize(n);  // Trim unused rows to avoid garbage data

  int d = neighbor[0].size();    // number of dimensions (e.g., 3 for RGB)
  Matrix cov(d, vector<double>(d, 0.0));
  computeCovarianceMatrix(neighbor, cov, n, d);
  float determ = determinant(cov); // cov is always 3*3, so we do not need to calculate other size of cov matrix
  // float determ = 3.0;

  // Get kernel size
  // If determinant is very small. It cause too large kernel size
  if (std::abs(determ) < 1e-5) {
    // std::cout << "determinant is too small. We use kSize = 3 instead of that.\n";
    return 3.0f;  // or any safe default kernel size
  }

  return factor / determ; 
}

float Gaussian(float x, float y, float sigma) {
  return std::exp(-(x * x + y * y) / (2.0f * sigma * sigma)) / (2.0f * 3.14159265f * sigma * sigma);
}

cv::Vec3b Gaussian_conv(const cv::Mat_<cv::Vec3b>& source, cv::Mat_<cv::Vec3b>& destination, int i, int j, float neighborSize, float sigma, float factor) {
  float kernelSize = calc_kSize(source, i, j, neighborSize, factor);
  // cout << "k_size" << kernelSize << endl;
  // float kernelSize = 3.0;
  int sizeDiv2 = static_cast<int>(kernelSize / 2.0);
  cv::Vec3f pixelValue(0.0f, 0.0f, 0.0f);
  cv::Vec3f colorSum(0.0f, 0.0f, 0.0f);
  float weightSum = 0.0f;
  int halfWidth = source.cols/2;

  for (int dy = -sizeDiv2; dy <= sizeDiv2; ++dy) {
      for (int dx = -sizeDiv2; dx <= sizeDiv2; ++dx) {
          int y = i + dy;
          int x = j + dx;
          // Taking care about edges and center.
          if (x < 0) {
            pixelValue = source(y, abs(x-1));
          } else if (x > source.cols) {
            pixelValue = source(y, source.cols-x+1);
          } else if (y < 0) {
            pixelValue = source(abs(y-1), x);
          } else if (y > source.rows) {
            pixelValue = source(source.rows-y+1, x);
          } else if (j <= halfWidth){
            if ( halfWidth-x >= 0){
              pixelValue = source(y, x);
            } else {
              pixelValue = source(y, halfWidth-x+1+halfWidth);
            }
          } else if (j > halfWidth){
            if ( x-halfWidth > 0){
              pixelValue = source(y, x);
            } else {
              pixelValue = source(y, halfWidth-x+1+halfWidth);
            }
          } else {
            pixelValue = source(y, x);
          }

          // int y = std::clamp(i + dy, 0, source.rows - 1);
          // int x = std::clamp(j + dx, 0, source.cols - 1);
          // pixelValue = source(y, x);
          float weight = Gaussian(dx, dy, sigma);
          weightSum += weight;
          for (int c = 0; c < 3; ++c)
              colorSum[c] += weight * pixelValue[c];
      }
  }

  cv::Vec3b result;
  for (int c = 0; c < 3; ++c){
      result[c] = cv::saturate_cast<uchar>(colorSum[c] / weightSum);
  }
  destination(i, j) = result;
}

int main( int argc, char** argv )
{

  cv::Mat_<cv::Vec3b> source = cv::imread ( argv[1], cv::IMREAD_COLOR);
  // cv::Mat_<cv::Vec3b> destination ( source.rows, source.cols );
  cv::Mat_<cv::Vec3b> destination(source.rows, source.cols, cv::Vec3b(0,0,0));

  cv::imshow("Source Image", source );

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 500;

  const float neighborSize = atof(argv[2]);//neighbor size
  const float sigma = atof(argv[3]);
  const float factor = atof(argv[4]);
  
  for (int it=0;it<iter;it++)
    {
      #pragma omp parallel for
      // #pragma omp parallel for collapse(2)
      for (int i=0;i<source.rows;i++)
      {
        cout << i << endl;
	      for (int j=0;j<source.cols;j++)
        {
          // Gaussian_conv(source, destination, i, j, neighborSize, sigma);
          Gaussian_conv(source, destination, i, j, neighborSize, sigma, factor);
        } 
      }
    }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;

  cv::imshow("Processed Image", destination );

  
  cout << "neighborSize " << argv[2] << endl;
  cout << "sigma " << argv[3] << endl;
  cout << "Source cols: " << source.cols << endl;
  cout << "Total time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count()/iter << " s" << endl;
  cout << "IPS: " << iter/diff.count() << endl;

  
  
  cv::waitKey();
  return 0;
}
