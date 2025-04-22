#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"

__device__ float3 getPixelSafe(const cv::cuda::PtrStep<float3>& img, int x, int y, int width, int height) {
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
    return img(y, x);
}

__device__ float determinant(const float cov[3][3]) {
    return
        cov[0][0] * (cov[1][1] * cov[2][2] - cov[1][2] * cov[2][1]) -
        cov[0][1] * (cov[1][0] * cov[2][2] - cov[1][2] * cov[2][0]) +
        cov[0][2] * (cov[1][0] * cov[2][1] - cov[1][1] * cov[2][0]);
}

// __device__ int permutationSign(const int* perm, int n) {
//     int inversions = 0;
//     for (int i = 0; i < n; ++i) {
//         for (int j = i + 1; j < n; ++j) {
//             if (perm[i] > perm[j]) {
//                 ++inversions;
//             }
//         }
//     }
//     return (inversions % 2 == 0) ? 1 : -1;
// }

// __device__ bool nextPermutation(int* perm, int n) {
//     // Find the largest index k such that perm[k] < perm[k + 1]
//     int k = -1;
//     for (int i = n - 2; i >= 0; --i) {
//         if (perm[i] < perm[i + 1]) {
//             k = i;
//             break;
//         }
//     }
//     if (k == -1) return false;

//     // Find the largest index l > k such that perm[k] < perm[l]
//     int l = -1;
//     for (int i = n - 1; i > k; --i) {
//         if (perm[k] < perm[i]) {
//             l = i;
//             break;
//         }
//     }

//     // Swap perm[k] and perm[l]
//     int temp = perm[k];
//     perm[k] = perm[l];
//     perm[l] = temp;

//     // Reverse perm[k+1..n-1]
//     for (int i = k + 1, j = n - 1; i < j; ++i, --j) {
//         int t = perm[i];
//         perm[i] = perm[j];
//         perm[j] = t;
//     }

//     return true;
// }

// __device__ float determinant_leibniz(const float A[3][3]) {
//     const int n = 3;
//     int perm[n] = {0, 1, 2};
//     float det = 0.0f;

//     do {
//         float term = 1.0f;
//         for (int i = 0; i < n; ++i) {
//             term *= A[i][perm[i]];
//         }
//         det += permutationSign(perm, n) * term;
//     } while (nextPermutation(perm, n));

//     return det;
// }

__device__ float calc_kSize_cuda(const cv::cuda::PtrStep<float3>& src,
                                 int rows, int cols, int i, int j, float neighborSize, float factor,
                                 float* neighbor, int maxNeighborSize
                                ) {
    const int neighborDiv2 = static_cast<int>(neighborSize / 2.0f);
    // float factor = 10.0f;

    // collect neighbors and compute mean
    float sum[3] = {0.0f, 0.0f, 0.0f};
    // const int nSize = neighborSize * neighborSize;
    // float neighbor[nSize][3];  
    // float neighbor[3][3];  // up to 7x7
    int n = 0;

    for (int dy = -neighborDiv2; dy <= neighborDiv2; ++dy) {
        for (int dx = -neighborDiv2; dx <= neighborDiv2; ++dx) {
            int y = i + dy;
            int x = j + dx;
            if (x >= 0 && x < cols && y >= 0 && y < rows) {
                int idx = n * 3;
                if (idx + 2 >= maxNeighborSize * 3) break; 
                float3 pixel = getPixelSafe(src, x, y, cols, rows);
                neighbor[idx + 0] = pixel.x;
                neighbor[idx + 1] = pixel.y;
                neighbor[idx + 2] = pixel.z;
                sum[0] += pixel.x;
                sum[1] += pixel.y;
                sum[2] += pixel.z;
                ++n;
            }
        }
    }

    if (n == 0) return 3.0f;

    // compute covariance matrix
    float mean[3] = {sum[0] / n, sum[1] / n, sum[2] / n};
    float cov[3][3] = {0};
    for (int m = 0; m < n; ++m) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                cov[j][k] += (neighbor[m * 3 + j] - mean[j]) * (neighbor[m * 3 + k] - mean[k]);
            }
        }
    }
    for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
            cov[j][k] /= (n - 1);
        }
    }

    float det = determinant(cov); // cov is always 3*3, so we do not need to calculate other size of cov matrix
    // float det = determinant_leibniz(cov);
    if (fabsf(det) < 1e-5f) {
        return 3.0f;
    }

    return factor / det;
}

__device__ float Gaussian(float x, float y, float sigma) {
    return expf(-(x * x + y * y) / (2.0f * sigma * sigma)) / (2.0f * 3.14159265f * sigma * sigma);
}

__device__ float3 Gaussian_conv(const cv::cuda::PtrStep<float3> source,
                                int cols, int rows, int i, int j, float neighborSize, float sigma, float factor,
                                float* neighbor, int maxNeighborSize)
{
    float kSize = calc_kSize_cuda(source, rows, cols, i, j, neighborSize, factor, neighbor, maxNeighborSize);
    // float kSize = 3.0;
    int halfSize = static_cast<int>(kSize / 2.0f);
    float3 colorSum = make_float3(0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;
    int halfWidth = cols / 2;
    float3 pixelValue;

    for (int dy = -halfSize; dy <= halfSize; ++dy) {
        for (int dx = -halfSize; dx <= halfSize; ++dx) {
            int y = i + dy;
            int x = j + dx;

            if (x < 0) {
                pixelValue = source(y, abs(x - 1));
            } else if (x > cols) {
                pixelValue = source(y, cols - x + 1);
            } else if (y < 0) {
                pixelValue = source(abs(y - 1), x);
            } else if (y > rows) {
                pixelValue = source(rows - y + 1, x);
            } else if (j <= halfWidth) {
                if (halfWidth - x >= 0) {
                    pixelValue = source(y, x);
                } else {
                    pixelValue = source(y, halfWidth - x + 1 + halfWidth);
                }
            } else if (j > halfWidth) {
                if (x - halfWidth > 0) {
                    pixelValue = source(y, x);
                } else {
                    pixelValue = source(y, halfWidth - x + 1 + halfWidth);
                }
            } else {
                pixelValue = source(y, x);
            }

            // Clamp coordinates to image borders
            // y = max(0, min(y, rows - 1));
            // x = max(0, min(x, cols - 1));

            // float3 pixel = source(y, x);

            float weight = Gaussian(dx, dy, sigma);
            weightSum += weight;
            // colorSum += weight * pixel;
            colorSum += weight * pixelValue;
        }
    }

    // Normalize
    colorSum /= weightSum;

    return colorSum;
}

__global__ void process(const cv::cuda::PtrStep<float3> src,
                        cv::cuda::PtrStep<float3> dst,
                        int rows, int cols, float neighborSize, float sigma, float factor,
                        float* d_neighborBuffer, int maxNeighborSize
                        ) 
{

    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    const int i = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= rows || j >= cols) return;
    // if ((j > 0) && (j < cols - 1) && (i < rows - 1) && (i > 0)) // Ensure the coordinate is in source
    // {
    float3 resultPixel;
    
    int threadIndex = i * cols + j;
    if (threadIndex >= rows * cols) return;
    float* neighborPtr = &d_neighborBuffer[threadIndex * maxNeighborSize * 3];
    if (neighborPtr == nullptr) return;
    resultPixel = Gaussian_conv(src, cols, rows, i, j, neighborSize, sigma, factor, neighborPtr, maxNeighborSize);

    // clamp(resultPixel, 0.0, 1.0);
    resultPixel.x = fminf(fmaxf(resultPixel.x, 0.0f), 1.0f);
    resultPixel.y = fminf(fmaxf(resultPixel.y, 0.0f), 1.0f);
    resultPixel.z = fminf(fmaxf(resultPixel.z, 0.0f), 1.0f);
    dst(i, j) = resultPixel;
    // }
}

int divUp(int a, int b) // Ensures CUDA grid dimensions are big enough.
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, float neighborSize, float sigma, float factor)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

    dst.create(src.size(), src.type());

    int maxNeighborCount = static_cast<int>(ceil(neighborSize) * ceil(neighborSize));
    int totalPixels = src.rows * src.cols;
    size_t bufferSize = totalPixels * maxNeighborCount * 3 * sizeof(float);
    std::cout << "Allocating neighbor buffer of size: " << bufferSize / (1024.0 * 1024.0) << " MB" << std::endl;
    float* d_neighborBuffer; 

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // std::cout << "rows: " << src.rows << ", cols: " << src.cols << std::endl;
    // std::cout << "maxNeighborCount: " << maxNeighborCount << std::endl;
    // std::cout << "totalPixels: " << totalPixels << std::endl;
    // std::cout << "allocating buffer for: " << (totalPixels * maxNeighborCount * 3 * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
    process<<<grid, block>>>(src, dst, src.rows, src.cols, neighborSize, sigma, factor, d_neighborBuffer, maxNeighborCount);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device sync failed: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaFree(d_neighborBuffer);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
    }
}
