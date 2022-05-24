#include <bits/stdc++.h>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "kernel_launch.h"
#include "visualization.h"
#include "BisenetTensorRT.h"

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
    auto visualizer = Visualizer();

    std::string rootPath = "/home/amrelsersy/PointPainting/data/KITTI/testing/image_2";
    for (const auto & entry : std::filesystem::directory_iterator(rootPath))
    {
        std::string imagePath = entry.path().string();
        cv::Mat image = cv::imread(imagePath, cv::ImreadModes::IMREAD_COLOR);
        std::cout << imagePath << " " << image.size() << std::endl;
        
        auto t1 = std::chrono::system_clock::now();
        // do inference
        cv::Mat semantic = bisenet.Inference(image);
        auto t2 = std::chrono::system_clock::now();
        std::cout << "Inference = " << std::chrono::duration<double>(t2-t1).count() * 1e3 << " ms" << std::endl;

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