#ifndef VISUALIZER
#define VISUALIZER

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
        semanticToColorMap[0] =  {128, 64,128};
        semanticToColorMap[1] =  {244, 35,232};
        semanticToColorMap[2] =  { 70, 70, 70};
        semanticToColorMap[3] =  {102,102,156};
        semanticToColorMap[4] =  {190,153,153};
        semanticToColorMap[5] =  {153,153,153};
        semanticToColorMap[6] =  {250,170, 30};
        semanticToColorMap[7] =  {220,220,  0};
        semanticToColorMap[8] =  {107,142, 35};
        semanticToColorMap[9] =  {152,251,152};
        semanticToColorMap[10] = { 70,130,180};
        semanticToColorMap[11] = {220, 20, 60};
        semanticToColorMap[12] = {255,  0,  0};
        semanticToColorMap[13] = {  0,  0,142};
        semanticToColorMap[14] = {  0,  0, 70};
        semanticToColorMap[15] = {  0, 60,100};
        semanticToColorMap[16] = {0, 80,100};
        semanticToColorMap[17] = {0,  0,230};
        semanticToColorMap[18] = {119, 11, 32};        
    }
    ~Visualizer()
    {
    }

    void ConvertToSemanticMap(cv::Mat image, cv::Mat &semantic)
    {
        // the visualized image type format is CV_8UC3 which is 3 channels of 8-bit unsinged char
        semantic = cv::Mat(this->shape, CV_8UC3);
        for (int row = 0; row < image.rows; row++)
        {
            for (int col = 0; col < image.cols; col++)
            {
                // get the semantic id, which is from 0-18
                uchar id = image.at<unsigned char>(row,col);
                // Vec3b is unsinged char of 3 channels 
                semantic.at<cv::Vec3b>(row,col) = this->semanticToColorMap[id];
            }
        }

        // just for visualization, convert to BGR
        cv::cvtColor(semantic, semantic, cv::COLOR_RGB2BGR);
    }

    std::vector<float> ConvertToSemanticPointcloud(std::vector<float> pointcloud)
    {
        return std::vector<float>();
    }

private:
    ///\brief map the semantic ids to its coresponding color in the colored semantic map
    std::map<uchar, cv::Vec3b> semanticToColorMap;
    ///\brief shape of the semantic map
    cv::Size shape;

};

#endif