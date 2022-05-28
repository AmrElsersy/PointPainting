#include <cuda_runtime.h>
#include <cudaProfiler.h>
#include <cuda_profiler_api.h>
#include <cuda.h>
#include "device_launch_parameters.h"

#define WIDTH 1024
#define HEIGHT 512
#define CHANNELS 19
#define DONT_CARE -1

void argmaxLaunchKernel(float *bisenet_output, unsigned char *argmax_output, cudaStream_t stream);


#define WIDTH_SEMANTIC_KITTI 1242
#define HEIGHT_SEMANTIC_KITTI 375
#define POINTCLOUD_CHANNELS 4
#define UNLABELED_POINT 255

void pointpainting(float *pointcloud, unsigned char *semantic_map, float *proj_matrix, int n_points, 
                    unsigned char *pointcloud_semantic, cudaStream_t stream);
