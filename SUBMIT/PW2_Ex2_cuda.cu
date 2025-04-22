#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"

__device__ float Gaussian(float x, float y, float sigma)
{
    return expf(-(x * x + y * y) / (2.0f * sigma * sigma)) / (2.0f * 3.14159265f * sigma * sigma);
}

__device__ float3 Gaussian_conv(const cv::cuda::PtrStep<float3> source, int cols, int rows, int i, int j, float kSize, float sigma)
{
    int halfSize = static_cast<int>(kSize / 2.0f);
    float3 colorSum = make_float3(0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;
    int halfWidth = cols / 2;
    float3 pixelValue;

    for (int dy = -halfSize; dy <= halfSize; ++dy)
    {
        for (int dx = -halfSize; dx <= halfSize; ++dx)
        {
            int y = i + dy;
            int x = j + dx;

            if (x < 0)
            {
                pixelValue = source(y, abs(x - 1));
            }
            else if (x > cols)
            {
                pixelValue = source(y, cols - x + 1);
            }
            else if (y < 0)
            {
                pixelValue = source(abs(y - 1), x);
            }
            else if (y > rows)
            {
                pixelValue = source(rows - y + 1, x);
            }
            else if (j <= halfWidth)
            {
                if (halfWidth - x >= 0)
                {
                    pixelValue = source(y, x);
                }
                else
                {
                    pixelValue = source(y, halfWidth - x + 1 + halfWidth);
                }
            }
            else if (j > halfWidth)
            {
                if (x - halfWidth > 0)
                {
                    pixelValue = source(y, x);
                }
                else
                {
                    pixelValue = source(y, halfWidth - x + 1 + halfWidth);
                }
            }
            else
            {
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

__global__ void process_shared(const cv::cuda::PtrStep<float3> src,
                               cv::cuda::PtrStep<float3> dst,
                               int rows, int cols, float kSize, float sigma)
{
    extern __shared__ float3 shared[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int j = blockDim.x * blockIdx.x + tx;
    int i = blockDim.y * blockIdx.y + ty;

    int sharedWidth = blockDim.x + (int)kSize - 1;
    int sharedHeight = blockDim.y + (int)kSize - 1;
    int radius = kSize / 2;

    // Compute global coordinates of the top-left corner of shared memory block
    int shared_i = ty + radius;
    int shared_j = tx + radius;

    // Load data into shared memory
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            int global_y = i + dy;
            int global_x = j + dx;
            global_y = min(max(global_y, 0), rows - 1);
            global_x = min(max(global_x, 0), cols - 1);

            int shared_y = shared_i + dy;
            int shared_x = shared_j + dx;
            shared[shared_y * sharedWidth + shared_x] = src(global_y, global_x);
        }
    }

    __syncthreads(); // wait every thredads

    if (j >= cols || i >= rows)
        return;

    float3 colorSum = make_float3(0, 0, 0);
    float weightSum = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            float weight = Gaussian(dx, dy, sigma);
            float3 val = shared[(shared_i + dy) * sharedWidth + (shared_j + dx)];
            colorSum += weight * val;
            weightSum += weight;
        }
    }

    colorSum /= weightSum;
    // clamp(colorSum, 0.0f, 1.0f);
    colorSum.x = fminf(fmaxf(colorSum.x, 0.0f), 1.0f);
    colorSum.y = fminf(fmaxf(colorSum.y, 0.0f), 1.0f);
    colorSum.z = fminf(fmaxf(colorSum.z, 0.0f), 1.0f);
    dst(i, j) = colorSum;
}

int divUp(int a, int b) // Ensures CUDA grid dimensions are big enough.
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, float kSize, float sigma)
{
    const dim3 block(16, 16);
    const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

    // size_t sharedMemSize = (block.x + (int)kSize - 1) * (block.y + (int)kSize - 1) * sizeof(float3);
    int radius = static_cast<int>(kSize / 2);
    int sharedWidth = block.x + 2 * radius;
    int sharedHeight = block.y + 2 * radius;
    size_t sharedMemSize = sharedWidth * sharedHeight * sizeof(float3);

    process_shared<<<grid, block, sharedMemSize>>>(src, dst, src.rows, src.cols, kSize, sigma);
}
