#include <cuda_runtime.h>
#include <cudaProfiler.h>
#include <cuda_profiler_api.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <math.h>

#include <bits/stdc++.h>

#include "kernel_launch.h"

#define BLOCK_THREADS 64

// input size = (3, 512, 1024)
// output size argmax is (512, 1024)
__global__ void argmax_cuda(float *bisenet_output, unsigned char *argmax_output, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N)
        return;

    // argmax_output[tid] = bisenet_output[tid];
    float max = -INFINITY;
    unsigned char arg = 255;

    for (unsigned char i = 0; i < CHANNELS; i++)
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

void argmaxLaunchKernel(float *bisenet_output, unsigned char *argmax_output, cudaStream_t stream)
{
    int N = WIDTH * HEIGHT;
    int threadsPerBlock = 256;
    int numBlocks = ceil(double(N) / threadsPerBlock);
    argmax_cuda<<<numBlocks, threadsPerBlock, 0, stream>>>(bisenet_output, argmax_output, N);
}

/*
__device__ void atomic_argmax(float val, unsigned int arg, 
                             unsigned int *argmax_address, float *max_address)
{
    // note that this access is from the shared memory
    float old_max = *max_address;
    unsigned int old_arg = *argmax_address;
    float assumed_max;

    do {
        old_max = *max_address;

        // don't waste your time if it is not the maximum
        if (old_max > val)
            return;

        old_max = *max_address;
        old_arg = *argmax_address;
        
        assumed_max = old_max;
        
        // FAILED ALGORITHM
        // Because both atomic operations are related to each other and must do together(assigning the argmax and its corresponding max)
        // It is not possible to garuntee that, as propably just one of them will happen without the other
        old_arg = atomicExch(argmax_address, arg);        
        old_max = atomicExch(max_address,    val);        
    } 
    while(assumed_max != old_max);
}

__global__ void argmax_cuda_atomic(float *bisenet_output, unsigned char *argmax_address)
{

    __shared__ unsigned int block_argmax[BLOCK_THREADS * BLOCK_THREADS];
    __shared__ float block_max[BLOCK_THREADS * BLOCK_THREADS];

    int col     = threadIdx.x + blockIdx.x * blockDim.x;
    int row     = threadIdx.y + blockIdx.y * blockDim.y;
    int channel = threadIdx.z + blockIdx.z * blockDim.z; // blockDim.z = 0 .. equivelent: channel = threadIdx.z

    if (row > HEIGHT || col > WIDTH)
        return;

    // initialize the maximum and argmax of the block
    // do that just for 1 channel (any channel), as shared memory is just 1 channel not 19 channel as input
    if (threadIdx.z == 0)
    {
        int shared_id = threadIdx.x * threadIdx.y;
        block_argmax[shared_id] = 255;
        block_max[shared_id] = -INFINITY;
    }
    __syncthreads();


    // indexing: row-wise serialization so row * width to get the start address of that row, +col to get a specefic column values(19 values)
    // + (channel * width * height) as input data is NCHW format, so each channel is stored first as (width*height) before the next (width*height) one
    int tid = row * WIDTH + col + (channel * WIDTH * HEIGHT);
    float channel_value = bisenet_output[tid];

    // index of the max & argmax that is common for the whole 19 channel for each position in the map of size width*height
    int common_index = row * WIDTH + col;

    // do atomic operation as all 19 channel will access the same addresses of the max & argmax, 
    //so they should repeatly(atomically) access that addresses
    atomic_argmax(channel_value, channel, &block_argmax[common_index], &block_max[common_index]);

    // wait till all threads compare its value with the maximum to store the right argmax
    __syncthreads();

    // only store the argmax output by one thread of the 19 channel
    if (threadIdx.z == 0)
        argmax_address[common_index] = (unsigned char)block_argmax[common_index];
}

void argmaxLaunchKernel(float *bisenet_output, unsigned char *argmax_output, cudaStream_t stream)
{
    int n_threads = BLOCK_THREADS;
    dim3 threadsPerBlock(n_threads, n_threads, CHANNELS);
    dim3 numBlocks(ceil(double(WIDTH) / threadsPerBlock.x), ceil(double(HEIGHT) / threadsPerBlock.y), 1);
    argmax_cuda_atomic<<<numBlocks, threadsPerBlock, 0, stream>>>(bisenet_output, argmax_output);
}

// https://on-demand.gputechconf.com/gtc/2013/presentations/S3101-Atomic-Memory-Operations.pdf

*/
/*
By running some performance experiments, optimal performance is
achieved when the number of blocks we launch is exactly twice the number of
multiprocessors our GPU contains. For example, a GeForce GTX 280 has 30 multiprocessors, so our histogram kernel happens to run fastest on a GeForce GTX 280
when launched with 60 parallel blocks
*/