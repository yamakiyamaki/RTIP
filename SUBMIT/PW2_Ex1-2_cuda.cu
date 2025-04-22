#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"

__device__ float Gaussian(float x, float y, float sigma) {
    return expf(-(x * x + y * y) / (2.0f * sigma * sigma)) / (2.0f * 3.14159265f * sigma * sigma);
}

__device__ float3 Gaussian_conv(const cv::cuda::PtrStep<float3> source, int cols, int rows, int i, int j, float kSize, float sigma)
{
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
                        int rows, int cols, float kSize, float sigma) 
{

    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    const int i = blockDim.y * blockIdx.y + threadIdx.y;

    if ((j > 0) && (j < cols - 1) && (i < rows - 1) && (i > 0)) // Ensure the coordinate is in source
    {
        float3 resultPixel;
        
        resultPixel = Gaussian_conv(src, cols, rows, i, j, kSize, sigma);

        // clamp(resultPixel, 0.0, 1.0);
        resultPixel.x = fminf(fmaxf(resultPixel.x, 0.0f), 1.0f);
        resultPixel.y = fminf(fmaxf(resultPixel.y, 0.0f), 1.0f);
        resultPixel.z = fminf(fmaxf(resultPixel.z, 0.0f), 1.0f);
        dst(i, j) = resultPixel;
    }
}

int divUp(int a, int b) // Ensures CUDA grid dimensions are big enough.
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, float kSize, float sigma)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

    process<<<grid, block>>>(src, dst, src.rows, src.cols, kSize, sigma);
}
