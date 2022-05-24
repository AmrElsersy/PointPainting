#include <cuda_runtime.h>
#include <cudaProfiler.h>
#include <cuda_profiler_api.h>
#include <cuda.h>
 #include "device_launch_parameters.h"
#include <math.h>

#include <bits/stdc++.h>

#include "kernel_launch.h"


// input size = (3, 512, 1024)
// output size argmax is (512, 1024)
__global__ void argmax_cuda(float *bisenet_output, int *argmax_output, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N)
        return;

    // argmax_output[tid] = bisenet_output[tid];
    float max = -INFINITY;
    int arg = -1;

    for (int i = 0; i < CHANNELS; i++)
    {
        int channel_i = bisenet_output[tid + i * (WIDTH * HEIGHT) ];
        if (channel_i > max)
        {
            max = channel_i;
            arg = i;
        }
    }

    argmax_output[tid] = arg;
}

void verify_result(float *dev_bisenet_output, int *dev_argmax)
{
    float *host_bisenet = new float[WIDTH * HEIGHT * CHANNELS];
    float *host_argmax = new int[WIDTH * HEIGHT];
    float *result = new float[WIDTH * HEIGHT];

    cudaMemcpy(host_bisenet, dev_bisenet_output, WIDTH * HEIGHT * CHANNELS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_argmax, dev_argmax, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i =0 ; i < WIDTH*HEIGHT; i++)
    {
        float max = -INFINITY;
        float arg = -1;

        for (int c = 0; c < CHANNELS; c++)
        {

        }
    }
}

void argmaxLaunchKernel(float *bisenet_output, int *argmax_output, cudaStream_t stream)
{
    int N = WIDTH * HEIGHT;
    int threadsPerBlock = 256;
    int numBlocks = ceil(double(N) / threadsPerBlock);

    argmax_cuda<<<numBlocks, threadsPerBlock, 0, stream>>>(bisenet_output, argmax_output, N);
}
