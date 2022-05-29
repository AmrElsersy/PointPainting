#include <bits/stdc++.h>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "kernel_launch.h"
#include "visualization.h"
#include "BisenetTensorRT.h"
#include "PointPainting.h"
#include <pointcloud_common.h>

cv::Mat global_image;
void mouseHandler(int event,int x,int y, int flags,void* param)
{
    if (event == cv::EVENT_LBUTTONDOWN)
        std::cout << x << ", " << y << "  =  " << (int)global_image.at<uchar>(y,x) << std::endl;
}

int main(int argc, char** argv)
{
    std::string enginePath = "/home/amrelsersy/PointPainting/bisenet_tensorrt.trt";
    if (argc > 1)
        enginePath = argv[1];

    auto bisenet = BiseNetTensorRT(enginePath);
    auto painter = PointPainter();
    auto visualizer = Visualizer();

    std::string savePointcloudsPath = "/home/amrelsersy/PointPainting/tensorrt_inference/results_pointclouds";
    std::string rootPath = "/home/amrelsersy/PointPainting/data/KITTI/kitti/";
    std::string rootImagesPath = rootPath + "image_2";
    std::string rootPointcloudsPath = rootPath + "velodyne";

    // read the images & pointclouds paths
    int n_samples = 0;
    for (const auto & entry : std::filesystem::directory_iterator(rootImagesPath))
        n_samples++;

    std::vector<std::pair<std::string, std::string>> paths(n_samples);

    int i =0;
    for (const auto & entry : std::filesystem::directory_iterator(rootImagesPath))
    {
        std::string imagePath = entry.path().string();
        paths[i].first = imagePath;
        i++;
    }
    i =0;
    for (const auto & entry : std::filesystem::directory_iterator(rootPointcloudsPath))
    {
        std::string pointcloudPath = entry.path().string();
        paths[i].second = pointcloudPath;
        i++;
    }

    // loop over samples
    for (auto samplePaths : paths)
    {
        // read image
        std::string imagePath = samplePaths.first;
        std::string pointcloudPath = samplePaths.second;

        cv::Mat image = cv::imread(imagePath, cv::ImreadModes::IMREAD_COLOR);
        std::vector<Point> pointcloud = loadPointCloud(pointcloudPath);

        std::cout << imagePath << " " << pointcloudPath << std::endl;
        auto t1 = std::chrono::system_clock::now();

        // do inference
        cv::Mat semantic = bisenet.Inference(image);

        auto t2 = std::chrono::system_clock::now();
        std::cout << "Inference = " << std::chrono::duration<double>(t2-t1).count() * 1e3 << " ms" << std::endl;

        // pointpainting
        painter.Paint(pointcloud, semantic);
        savePointCloud(pointcloud, savePointcloudsPath);

        // visualization
        cv::Mat coloredSemantic;
        visualizer.ConvertToSemanticMap(semantic, coloredSemantic);
        cv::imshow("image", image);
        cv::imshow("coloredSemantic", coloredSemantic);
        global_image = semantic;
        cv::setMouseCallback("coloredSemantic", mouseHandler, &coloredSemantic);
        // just for visualization ,add 20 to brightness the image, as ids is [0-19] which is really dark
        // semantic.convertTo(semantic,CV_8UC1, 1, 20);
        // cv::imshow("semantic", semantic);
        if (cv::waitKey(0) == 27)
        {
            cv::destroyAllWindows();
            return 0;
        }
    }
    return 0;
}