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
    float P2[16];
    float rect[16];
    float veloToCam[16];
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

public:
    PointPainter()
    {
        // sizes
        this->sizeSemanticBytes = WIDTH_SEMANTIC_KITTI * HEIGHT_SEMANTIC_KITTI * sizeof(unsigned char);
        this->sizePointcloudBytes = MAX_NUM_POINTS_IN_POINTCLOUD * POINTCLOUD_CHANNELS * sizeof(float);
        this->sizePointcloudSemanticBytes = MAX_NUM_POINTS_IN_POINTCLOUD * sizeof(unsigned char);

        // calculate projection matrix

        // allocate device memory
        this->AllocateDeviceMemories();

        // allocate max memory for the semantic cloud
        this->pointcloud_semantic.resize(MAX_NUM_POINTS_IN_POINTCLOUD);
    }
    ~PointPainter()
    {
        cudaFree(this->dev_pointcloud);
        cudaFree(this->dev_pointcloud_semantic);
        cudaFree(this->dev_semantic);
    }

    void Paint(std::vector<Point> &pointcloud, cv::Mat &semantic)
    {
        // copy pointcloud to GPU memory
        float *host_pointcloud = convertPointsToArray(pointcloud).data();
        cudaMemcpy(this->dev_pointcloud, host_pointcloud, pointcloud.size() * POINTCLOUD_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);

        // NOTE:::: COPY FIRST TO HOST PAGE LOCKED MEMORY THEN TO GPU 
        // convert semantic map from cv::Mat to array and allocate memory for semantic map
        cudaMemcpy(this->dev_semantic, semantic.data, this->sizeSemanticBytes, cudaMemcpyHostToDevice); // THIS IS ONLY FROM PINNED CPU TO GPU

        // painting
        pointpainting(dev_pointcloud, dev_semantic, projectionMatrix, pointcloud.size(), dev_pointcloud_semantic);

        // copy result to cpu
        cudaMemcpy(this->pointcloud_semantic.data(), this->dev_pointcloud_semantic, pointcloud.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // replace intensity with paining (for visualization)
        for (int i = 0; i < pointcloud.size(); i++)
        {
            auto point = pointcloud[i];
            point.intensity = this->pointcloud_semantic[i];
        }
    }

    void AllocateDeviceMemories()
    {
        cudaMalloc(this->dev_semantic, this->sizeSemanticBytes);
        cudaMalloc(this->dev_pointcloud, this->sizePointcloudBytes);
        cudaMalloc(this->dev_pointcloud_semantic, this->sizePointcloudSemanticBytes);
    }

};