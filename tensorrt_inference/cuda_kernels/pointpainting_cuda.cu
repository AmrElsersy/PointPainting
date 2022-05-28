#include "kernel_launch.h"
#include <math.h>


__constant__ float projection_matrix[16];

__global__ void painting_kernel(float *pointcloud, unsigned char *semantic_map, unsigned char* pointcloud_semantic, int n_points)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _points)
        return;

    float4 point = pointcloud[tid * POINTCLOUD_CHANNELS];
    point.w = 1;

    float projected_point[4];

    // transform each point with the projection matrix (proj is the dot product of the 3 matrices : P2 @ rect @ velo_to_cam)
    for (int i = 0; i < 4; i++)
    {
        float dotProduct = 0;
        dotProduct += projection_matrix[i * 4 + 0] * point.x;
        dotProduct += projection_matrix[i * 4 + 1] * point.y;
        dotProduct += projection_matrix[i * 4 + 2] * point.z;
        dotProduct += projection_matrix[i * 4 + 3] * point.w;

        projected_point[i] = dotProduct;        
    }

    // devide by homogenious part (devide by z)
    projected_point[0] /= projected_point[2];
    projected_point[1] /= projected_point[2];

    // get x,y coordinates of the semantic map
    int x_semantic = (int)projected_point[0]; 
    int y_semantic = (int)projected_point[1];

    // only assign a label to the point if its projected point lies inside the semantic map, otherwise it is already has unlabeled value(255)
    if (x_semantic < HEIGHT_SEMANTIC_KITTI && y_semantic < WIDTH_SEMANTIC_KITTI)
        // assign a label to the point with the corresponding x,y projected point in the semantic map
        pointcloud_semantic[tid] = semantic_map[x_semantic * WIDTH_SEMANTIC_KITTI + y_semantic];
}

void pointpainting(float *pointcloud, unsigned char *semantic_map, float *proj_matrix, int n_points, 
                    unsigned char *pointcloud_semantic, cudaStream_t stream)
{
    // set the constant projection matrix
    cudaMemcpyToSymbol(projection_matrix, proj_matrix, 16 * sizeof(float), 0UL, cudaMemcpyHostToDevice);

    // set all points to be unlabeled till we label them
    cudaMemset(pointcloud_semantic, UNLABELED_POINT, n_points * sizeof(unsigned char));

    // device multiprocessors
    // int device;
    // cudaDeviceGet

    int threadsPerBlock = 128;
    int numBlocks = ceil(double(n_points) / threadsPerBlock);
    painting_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(pointcloud, semantic_map, pointcloud_semantic, n_points);
}