#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <kernel_launch.h>

class Visualizer
{
public: 
    Visualizer(cv::Size _shape = cv::Size(1024, 512))
    {
        shape = _shape;
        this->semanticMap = cv::Mat(this->shape, CV_8UC3);

        semanticToColorMap[0] =  {255, 255, 255};
        semanticToColorMap[1] =  {255, 0, 0};
        semanticToColorMap[2] =  {0, 255, 0};
        semanticToColorMap[3] =  {0, 0, 255};
        semanticToColorMap[4] =  {255, 255, 0};
        semanticToColorMap[5] =  {0, 255, 255};
        semanticToColorMap[6] =  {255, 0, 255};
        semanticToColorMap[7] =  {150, 125, 0};
        semanticToColorMap[8] =  {0, 100, 255};
        semanticToColorMap[9] =  {200, 230, 0};
        semanticToColorMap[10] = {200, 40, 30};
        semanticToColorMap[11] = {20, 189, 0};
        semanticToColorMap[12] = {0, 200, 150};
        semanticToColorMap[13] = {30, 50, 213};
        semanticToColorMap[14] = {120, 60, 70};
        semanticToColorMap[15] = {0, 200, 50};
        semanticToColorMap[16] = {60, 123, 50};
        semanticToColorMap[17] = {90, 90, 120};
        semanticToColorMap[18] = {255, 255, 255};        
        semanticToColorMap[DONT_CARE] = {0, 0, 0};        
    }
    ~Visualizer()
    {
        std::cout << "Destructor Visualizer " << std::endl;
    }

    void ConvertToSemanticMap(cv::Mat image, cv::Mat &semantic)
    {
        semantic = cv::Mat(this->shape, CV_8UC3, image.data);
        return;
        std::cout << "SEMANTICCCC " << semanticToColorMap[15] << std::endl;
        semantic = this->semanticMap;
        for (int row = 0; row < image.rows; row++)
        {
            for (int col = 0; col < image.cols; col++)
            {
                // std::cout << row << ", " << col << std::endl; 
                float id = image.at<float>(row,col);
                std::cout << " id =" << id << std::endl;
                semantic.at<cv::Vec3i>(row,col) = this->semanticToColorMap[id];
                // std::cout << "$$" << std::endl;
            }
        }
    }

    std::vector<float> ConvertToSemanticPointcloud(std::vector<float> pointcloud)
    {
        return std::vector<float>();
    }

private:
    std::map<int, cv::Vec3i> semanticToColorMap;
    cv::Size shape;
    cv::Mat semanticMap;

};