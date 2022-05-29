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

    auto painter = PointPainter();
    auto bisenet = BiseNetTensorRT(enginePath);
    auto visualizer = Visualizer();

    std::string savePointcloudsPath = "/home/amrelsersy/PointPainting/tensorrt_inference/results_pointclouds/";
    std::string rootPath = "/home/amrelsersy/PointPainting/data/KITTI/kitti/";
    std::string rootImagesPath = rootPath + "image_2";
    std::string rootPointcloudsPath = rootPath + "velodyne";

    // read the images & pointclouds paths
    std::vector<std::string> imagesPaths;
    std::vector<std::string> pointcloudsPaths;
    
    for (const auto & entry : std::filesystem::directory_iterator(rootImagesPath))
    {
        std::string imagePath = entry.path().string();
        imagesPaths.push_back(imagePath);
    }

    for (const auto & entry : std::filesystem::directory_iterator(rootPointcloudsPath))
    {
        std::string pointcloudPath = entry.path().string();
        pointcloudsPaths.push_back(pointcloudPath);
    }

    std::sort(imagesPaths.begin(), imagesPaths.end());
    std::sort(pointcloudsPaths.begin(), pointcloudsPaths.end());
    

    // loop over samples
    for (int i = 0; i < imagesPaths.size(); i++)
    {
        // read image
        std::string imagePath = imagesPaths[i];
        std::string pointcloudPath = pointcloudsPaths[i];

        cv::Mat image = cv::imread(imagePath, cv::ImreadModes::IMREAD_COLOR);
        std::vector<Point> pointcloud = loadPointCloud(pointcloudPath);

        std::cout << imagePath << " " << pointcloudPath << std::endl;
        auto t1 = std::chrono::system_clock::now();

        // do inference
        cv::Mat semantic = bisenet.Inference(image);

        auto t2 = std::chrono::system_clock::now();
        std::cout << "Inference = " << std::chrono::duration<double>(t2-t1).count() * 1e3 << " ms" << std::endl;

        using namespace std;
        // pointpainting
        // painter.Paint(pointcloud, semantic);

        // for (auto point : pointcloud)
        // {
        //     if (point.intensity != 0)
        //         cout << point.intensity << " ";
        // }

        // savePointCloud(pointcloud, savePointcloudsPath + to_string(i) + ".bin");

        std::cout << semantic.size() << std::endl;
        // visualization

        cv::Mat coloredSemantic;
        visualizer.ConvertToSemanticMap(semantic, coloredSemantic);

        cv::imshow("image", image);
        cv::imshow("coloredSemantic", coloredSemantic);

        // global_image = cv::Mat(semantic);
        // cv::setMouseCallback("coloredSemantic", mouseHandler, &coloredSemantic);
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