#include "kernel_launch.h"
#include "pointcloud_common.h"
#include <bits/stdc++.h>
#include <eigen3/Eigen/Eigen>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

class PointPainter
{
private:
    float P2[16];
    float rect[16];
    float veloToCam[16];
    float projectionMatrix[16];

public:
    PointPainter()
    {
        // calculate projection matrix
    }
    ~PointPainter()
    {

    }

    std::vector<Point> Paint(std::vector<Point> pointcloud, cv::Mat semantic)
    {

    }

};