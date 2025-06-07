#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"

__device__ float3 TrueAnaglyphs(const cv::cuda::PtrStep<float3> source, int cols, int i, int j)
{
    float3 pixel_l = source(i, j);
    float3 pixel_r = source(i, j + cols / 2);

    float3 m = make_float3(0.299f, 0.587f, 0.114f);

    float3 anaglyphsValue;
    anaglyphsValue.z = m.x * pixel_l.z + m.y * pixel_l.y + m.z * pixel_l.x; // Red
    anaglyphsValue.y = 0.0f;                                                  // Green
    anaglyphsValue.x = m.x * pixel_r.z + m.y * pixel_r.y + m.z * pixel_r.x;   // Blue

    return anaglyphsValue;
}

__device__ float3 GrayAnaglyphs(const cv::cuda::PtrStep<float3> source, int cols, int i, int j)
{
    float3 pixel_l = source(i, j);
    float3 pixel_r = source(i, j + cols / 2);

    float3 m = make_float3(0.299f, 0.587f, 0.114f);

    float3 anaglyphsValue;
    anaglyphsValue.z = m.x * pixel_l.z + m.y * pixel_l.y + m.z * pixel_l.x; // Red
    anaglyphsValue.y = m.x * pixel_r.z + m.y * pixel_r.y + m.z * pixel_r.x; // Green                                                 // Green
    anaglyphsValue.x = m.x * pixel_r.z + m.y * pixel_r.y + m.z * pixel_r.x;   // Blue

    return anaglyphsValue;
}

__device__ float3 ColorAnaglyphs(const cv::cuda::PtrStep<float3> source, int cols, int i, int j)
{
    float3 pixel_l = source(i, j);
    float3 pixel_r = source(i, j + cols / 2);

    float3 anaglyphsValue;
    anaglyphsValue.z = 1.0f * pixel_l.z + 0.0f * pixel_l.y + 0.0f * pixel_l.x; // Red
    anaglyphsValue.y = 0.0f * pixel_r.z + 1.0f * pixel_r.y + 0.0f * pixel_r.x; // Green                                                 // Green
    anaglyphsValue.x = 0.0f * pixel_r.z + 0.0f * pixel_r.y + 1.0f * pixel_r.x;   // Blue

    return anaglyphsValue;
}

__device__ float3 HalfColorAnaglyphs(const cv::cuda::PtrStep<float3> source, int cols, int i, int j)
{
    float3 pixel_l = source(i, j);
    float3 pixel_r = source(i, j + cols / 2);

    float3 m = make_float3(0.299f, 0.587f, 0.114f);

    float3 anaglyphsValue;
    anaglyphsValue.z = m.x * pixel_l.z + m.y * pixel_l.y + m.z * pixel_l.x; // Red
    anaglyphsValue.y = 0.0f * pixel_r.z + 1.0f * pixel_r.y + 0.0f * pixel_r.x; // Green                                                 // Green
    anaglyphsValue.x = 0.0f * pixel_r.z + 0.0f * pixel_r.y + 1.0f * pixel_r.x;   // Blue

    return anaglyphsValue;
}

__device__ float3 OptimizedAnaglyphs(const cv::cuda::PtrStep<float3> source, int cols, int i, int j)
{
    float3 pixel_l = source(i, j);
    float3 pixel_r = source(i, j + cols / 2);

    float3 anaglyphsValue;
    anaglyphsValue.z = 0.0f * pixel_l.z + 0.7f * pixel_l.y + 0.3f * pixel_l.x; // Red
    anaglyphsValue.y = 0.0f * pixel_r.z + 1.0f * pixel_r.y + 0.0f * pixel_r.x; // Green                                                 // Green
    anaglyphsValue.x = 0.0f * pixel_r.z + 0.0f * pixel_r.y + 1.0f * pixel_r.x;   // Blue

    return anaglyphsValue;
}

__global__ void process(const cv::cuda::PtrStep<float3> src,
                        cv::cuda::PtrStep<float3> dst,
                        int rows, int cols, int modeN)
{

    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    const int i = blockDim.y * blockIdx.y + threadIdx.y;

    if ((j > 0) && (j < cols/2 - 1) && (i < rows - 1) && (i > 0)) // Ensure the coordinate is in source
    {
        float3 resultPixel;
        if (modeN == 0)
        {
            resultPixel = TrueAnaglyphs(src, cols, i, j);
        }
        else if (modeN == 1) {
            resultPixel = GrayAnaglyphs(src, cols, i, j );
        }
        else if (modeN == 2) {
            resultPixel = ColorAnaglyphs(src, cols, i, j );
        }
        else if (modeN == 3) {
            resultPixel = HalfColorAnaglyphs(src, cols, i, j );
        }
        else if (modeN == 4) {
            resultPixel = OptimizedAnaglyphs(src, cols, i, j );
        }

        clamp(resultPixel, 0.0, 1.0);
        dst(i, j) = resultPixel;
    }
}

int divUp(int a, int b) // Ensures CUDA grid dimensions are big enough.
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, int modeN, int blockLines, int blockCols)
{
    // const dim3 block(32,8);
    const dim3 block(blockLines,blockCols);
    const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

    process<<<grid, block>>>(src, dst, src.rows, src.cols, modeN);
}
