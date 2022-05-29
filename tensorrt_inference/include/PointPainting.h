#include "kernel_launch.h"
#include "pointcloud_common.h"
#include <bits/stdc++.h>
#include <eigen3/Eigen/Eigen>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#define MAX_NUM_POINTS_IN_POINTCLOUD 120 * 1000

class PointPainter
{
private:
    // calib matrices
    float projectionMatrix[16];

    // Device memory
    float *dev_pointcloud;
    unsigned char *dev_semantic;
    unsigned char *dev_pointcloud_semantic;

    // Host memory
    std::vector<unsigned char> pointcloud_semantic;
    float *host_pointcloud;
    unsigned char *host_semantic;
    unsigned char *host_pointcloud_semantic;

    // sizes
    int sizeSemanticBytes;
    int sizePointcloudBytes;
    int sizePointcloudSemanticBytes;
    int sizeInputPointsBytes;

    // stream
    cudaStream_t stream;

public:
    PointPainter()
    {
        // sizes
        this->sizeSemanticBytes = WIDTH_SEMANTIC_KITTI * HEIGHT_SEMANTIC_KITTI * sizeof(unsigned char);
        this->sizePointcloudBytes = MAX_NUM_POINTS_IN_POINTCLOUD * POINTCLOUD_CHANNELS * sizeof(float);
        this->sizePointcloudSemanticBytes = MAX_NUM_POINTS_IN_POINTCLOUD * sizeof(unsigned char);

        // calculate projection matrix
        this->SetProjectionMatric();

        // allocate host & device memory
        this->AllocateInferenceMemory();

        // allocate max memory for the semantic cloud
        this->pointcloud_semantic.resize(MAX_NUM_POINTS_IN_POINTCLOUD);

        // create stream for synchronization
        cudaStreamCreate(&this->stream);
    }
    ~PointPainter()
    {
        cudaFree(this->dev_pointcloud);
        cudaFree(this->dev_pointcloud_semantic);
        cudaFree(this->dev_semantic);
        cudaFree(this->host_pointcloud);
        cudaFree(this->host_pointcloud_semantic);
        cudaFree(this->host_semantic);
    }

    void Paint(std::vector<Point> &pointcloud, cv::Mat &semantic)
    {
        // calculate the size  of the actual points in the pointcloud(not the maximum bytes allocated for inference)
        // used in inference functions
        this->sizeInputPointsBytes = pointcloud.size() * POINTCLOUD_CHANNELS * sizeof(float);

        // convert pointcloud to array
        float *pointcloud_data = convertPointsToArray(pointcloud).data();

        // resize the semantic shape[1024, 512] to standard kitti shape[1242, 375]
        cv::resize(semantic, semantic, cv::Size(WIDTH_SEMANTIC_KITTI, HEIGHT_SEMANTIC_KITTI));

        // copy to page-locked memory then to GPU memory
        memcpy(this->host_pointcloud, pointcloud_data, sizeInputPointsBytes);

        // convert semantic map from cv::Mat to array and copy it to the page-locked memory
        memcpy(this->host_semantic, semantic.data, this->sizeSemanticBytes);

        this->Inference(pointcloud.size());
        // this->InferenceAsyncV2(pointcloud.size());

        // copy from page-locked memory to cpu memory
        memcpy(this->pointcloud_semantic.data(), this->host_pointcloud_semantic, pointcloud.size() * sizeof(float));

        // replace intensity with paining (for visualization)
        for (int i = 0; i < pointcloud.size(); i++)
        {
            auto point = pointcloud[i];
            point.intensity = this->pointcloud_semantic[i];
            // access the pinned page-locked memory directly 
            // point.intensity = this->host_pointcloud_semantic[i];
        }

        free(pointcloud_data);
    }

    void Inference(int n_points)
    {
        // copy pointcloud & semantic map from page-locked memory to device memory
        cudaMemcpy(this->dev_pointcloud, this->host_pointcloud, this->sizeInputPointsBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->dev_semantic, this->host_semantic, this->sizeSemanticBytes, cudaMemcpyHostToDevice);

        // painting
        pointpainting(dev_pointcloud, dev_semantic, projectionMatrix, n_points, dev_pointcloud_semantic, cudaStreamDefault);

        // copy result to page-locked memory
        cudaMemcpy(this->host_pointcloud_semantic, this->dev_pointcloud_semantic, 
            n_points * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void InferenceAsyncV2(int n_points)
    {
        // Async copy from cpu to page-locked memory for pointcloud & semantic map
        cudaMemcpyAsync(this->dev_pointcloud, this->host_pointcloud, this->sizeInputPointsBytes, cudaMemcpyHostToDevice, this->stream);
        cudaMemcpyAsync(this->dev_semantic, this->host_semantic, this->sizeSemanticBytes, cudaMemcpyHostToDevice, this->stream);

        // painting
        pointpainting(dev_pointcloud, dev_semantic, projectionMatrix, n_points, dev_pointcloud_semantic, this->stream);

        // copy result to page-locked memory
        cudaMemcpyAsync(this->host_pointcloud_semantic, this->dev_pointcloud_semantic, 
            n_points * sizeof(float), cudaMemcpyDeviceToHost, this->stream);

        // Synchronize, wait till all async operations finish
        cudaStreamSynchronize(this->stream);
    }

    void AllocateInferenceMemory()
    {
        // Device Memory
        cudaMalloc(this->dev_semantic, this->sizeSemanticBytes);
        cudaMalloc(this->dev_pointcloud, this->sizePointcloudBytes);
        cudaMalloc(this->dev_pointcloud_semantic, this->sizePointcloudSemanticBytes);

        // Page-Locked Host Memory
        cudaMallocHost(this->host_semantic, this->sizeSemanticBytes);
        cudaMallocHost(this->host_pointcloud, this->sizePointcloudBytes);
        cudaMallocHost(this->host_pointcloud_semantic, this->sizePointcloudSemanticBytes);
    }
    
    void SetProjectionMatrix()
    {
        // row based 4x4 projection matrix (it is 3x4 in kitti dataset with the last identity row to make it 4x4)
        // if you used your own setup, set that matrix with your camera's projection matrix parameters
        float P2_data[16] = {7.215377e+02, 0, 6.095593e+02, 4.485728e+01, 
                             0, 7.215377e+02, 1.72854e+02, 2.163791e-01, 
                             0, 0, 1.0, 2.745884e-03,
                             0, 0, 0, 1};

        // row based 4x4 rectification matrix (it is 3x3 matrix in kitti dataset with 0 translation in the last column)
        // rectification matrix is for kitti, as kitti has many cameras, if you have 1 camera, set that to identity matrix
        float rect_data[16] = {9.999239e-01, 9.83776e-03, -7.445048e-03, 0,
                               -9.869795e-03, 9.999421e-01, -4.278459e-03, 0,
                                7.402527e-03, 4.351614e-03, 9.999631e-01, 0,
                                0, 0, 0, 1};

        // row based 4x4 transformation matrix from velodyne coord to 3d camera coord (it is 3x4 mattrix in kitti with last row to make it 4x4)
        // if you used your own data, you must set that matrix with your own transformation matrix
        float velo_to_cam_data[16] = {7.533745e-03, -9.999714e-01, -6.16602e-04, -4.069766e-03, 
                                      1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02, 
                                      9.998621e-01, 7.52379e-03, 1.480755e-02, -2.717806e-01, 
                                      0, 0, 0, 1};

        Eigen::Matrix<float, 4, 4, Eigen::RowMajor> P2(P2_data);
        Eigen::Matrix<float, 4, 4, Eigen::RowMajor> rect(rect_data);
        Eigen::Matrix<float, 4, 4, Eigen::RowMajor> velo_to_cam(velo_to_cam_data);

        Eigen::Matrix<float, 4, 4, Eigen::RowMajor> final_matrix = P2 * rect * velo_to_cam;
        for (int i = 0; i < 16; i++)
            this->projectionMatrix[i] = final_matrix.data[i];
    }
};