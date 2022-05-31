#include <bits/stdc++.h>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "visualization.h"
#include "BisenetTensorRT.h"
#include "PointPainting.h"

int main(int argc, char** argv)
{
    std::string enginePath = "/home/amrelsersy/PointPainting/bisenet_tensorrt.trt";
    if (argc > 1)
        enginePath = argv[1];

    auto bisenet = BiseNetTensorRT(enginePath);
    auto painter = PointPainter();
    auto visualizer = Visualizer();

    std::string rootPath = "../data/";
    std::string savePointcloudsPath = "../data/results_pointclouds/";
    std::string rootImagesPath = rootPath + "image_2";
    std::string rootPointcloudsPath = rootPath + "velodyne";

    // read the images & pointclouds paths
    std::vector<std::string> imagesPaths;
    std::vector<std::string> pointcloudsPaths;
    
    for (const auto & entry : std::filesystem::directory_iterator(rootImagesPath))
        imagesPaths.push_back(entry.path().string());
    for (const auto & entry : std::filesystem::directory_iterator(rootPointcloudsPath))
        pointcloudsPaths.push_back(entry.path().string());

    std::sort(imagesPaths.begin(), imagesPaths.end());
    std::sort(pointcloudsPaths.begin(), pointcloudsPaths.end());

    // loop over samples
    for (int i = 0; i < imagesPaths.size(); i++)
    {
        // read image
        std::string imagePath = imagesPaths[i];
        cv::Mat image = cv::imread(imagePath, cv::ImreadModes::IMREAD_COLOR);

        // read pointcloud
        std::string pointcloudPath = pointcloudsPaths[i];
        std::vector<Point> pointcloud = loadPointCloud(pointcloudPath);

        auto t1 = std::chrono::system_clock::now();

        // do inference
        cv::Mat semantic = bisenet.Inference(image);

        auto t2 = std::chrono::system_clock::now();

        // pointpainting
        painter.Paint(pointcloud, semantic);

        auto t3 = std::chrono::system_clock::now();

        std::cout << "Bisenetv2 = " << std::chrono::duration<double>(t2-t1).count() * 1e3 << " ms" << std::endl;
        std::cout << "PointPainting = " << std::chrono::duration<double>(t3-t2).count() * 1e3 << " ms" << std::endl;
        std::cout << "Total Inference = " << std::chrono::duration<double>(t3-t1).count() * 1e3 << " ms" << std::endl;

        savePointCloud(pointcloud, savePointcloudsPath + to_string(i) + ".bin");

        // visualization
        cv::Mat coloredSemantic;
        visualizer.ConvertToSemanticMap(semantic, coloredSemantic);
        cv::imshow("coloredSemantic", coloredSemantic);

        // cv::imshow("image", image);
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