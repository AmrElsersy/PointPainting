#include <cuda_runtime.h>
#include <cudaProfiler.h>
#include <cuda_profiler_api.h>
#include <cuda.h>
#include "argmax_cuda.h"


__global__ void argmax_cuda(float *bisenet_output, float *argmax_output)
{

}

void argmax(float *bisenet_output, float *argmax_output)
{
    argmax_cuda<<<1,1>>>(bisenet_output, argmax_output);
}