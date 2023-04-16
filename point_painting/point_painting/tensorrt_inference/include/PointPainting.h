#ifndef POINT_PAINTING
#define POINT_PAINTING

#include <bits/stdc++.h>
#include <eigen3/Eigen/Eigen>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "pointcloud_io.h"
#include "kernel_launch.h"

class PointPainter
{
    ///\brief Final projection matrix = Projection @ Rectification @ Velo_to_Cam, used to project the 3D point of a pointcloud to the image
    /// It is a 4x4 row-order matrix (16 value stored [row1 row2 row3 row4]) 
    private: vector<float> projectionMatrix;

    ///\brief GPU Device memory for input pointcloud raw data
    private: float *dev_pointcloud;

    ///\brief GPU Device memory for input semantic image data
    private: unsigned char *dev_semantic;

    ///\brief GPU Device memory for output pointcloud semantic channel
    private: unsigned char *dev_pointcloud_semantic;

    ///\brief Host page-locked memory for input pointcloud
    private: float *host_pointcloud;

    ///\brief Host page-locked memory for input semantic image
    private: unsigned char *host_semantic;

    ///\brief Host page-locked memory for output pointcloud semantic channel
    private: unsigned char *host_pointcloud_semantic;

    ///\brief Size of semantic map in bytes 
    private: int sizeSemanticBytes;

    ///\brief Size of pointcloud raw data in bytes, which is maximum num of points * num of channels * sizeof(point)
    private: int sizePointcloudBytes;

    ///\brief Size of semantic channel of the pointcloud, which is num of points * size of(semantic value)
    private: int sizePointcloudSemanticBytes;

    ///\brief Size of pointcloud raw data in bytes, which is actual num of points * num of channels * sizeof(point)
    private: int sizeInputPointsBytes;

    ///\brief Cuda stream for synchronization
    private: cudaStream_t stream;

    ///\brief opencv image for resizing the semantic input    
    private: cv::Mat resizedSemantic;

    ///\brief Constructor
    public: PointPainter();

    ///\brief Destructor
    public: ~PointPainter();

    ///\brief Painting the pointcloud, label each point with a corresponding label by back-projeckting the semantic map image to the pointcloud
    /// & replace the intensity channel with a semantic label channel
    ///\param[inout] pointcloud input pointcloud to paint it with semantic labels ... intensity channel will be replaced with semantic values
    ///\param[in] semantic opencv image of the semantic map that has each pixel with a cooresponding label id of that object (ids [0-19])
    public: void Paint(std::vector<Point> &pointcloud, cv::Mat &semantic);

    ///\brief Synchronize Inference used by Paint() responsible for all cuda memory allocation & kernel launch
    ///\param[in] n_points num of points in the input pointcloud
    private: void Inference(int n_points);

    ///\brief Asynchronize Inference(faster) used by Paint() responsible for all cuda memory allocation & kernel launch
    ///\param[in] n_points num of points in the input pointcloud
    private: void InferenceAsyncV2(int n_points);

    ///\brief Allocate memory for cuda inference, memory includes GPU device memrories & host page-locked memories
    private: void AllocateInferenceMemory();

    ///\brief Set the values of the final projection matrix by multplying projection @ rectification @ velo_to_cam matrices of KITTI
    /// If you used your own data other than kitti, fill the values of the matrices in that function with your matrices values
    private: void SetProjectionMatrix();
};

#endif