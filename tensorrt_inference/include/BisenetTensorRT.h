#ifndef BISENET_TENSORRT
#define BISENET_TENSORRT

#include <bits/stdc++.h>
#include <fstream>
#include <NvInfer.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "kernel_launch.h"

#define INPUT_BINDING_INDEX 0
#define OUTPUT_BINDING_INDEX 1

class BiseNetTensorRT
{
    public: BiseNetTensorRT(std::string _enginePath);
    public: ~BiseNetTensorRT();
    public: cv::Mat Inference(cv::Mat image);
    
    //TensorRT requires your image data to be in NCHW order. But OpenCV reads it in NHWC order.
    void hwc_to_chw(cv::InputArray src, cv::OutputArray &dst);
    public: void PreProcessing(cv::Mat _image);

    private: void AllocateMemory();

    private: nvinfer1::IRuntime *runtime;
    private: nvinfer1::ICudaEngine *cudaEngine;
    private: nvinfer1::IExecutionContext *context;

    // Bindings buffers of device for Cuda context execution
    private: std::vector<void *> bindingBuffers;

    // Page-lock and GPU device memory for input & output
    private: float *hostInputMemory = nullptr;
    private: float *deviceInputMemory = nullptr;
    private: float *deviceOutputMemory = nullptr;
    private: uchar *hostOutputMemory = nullptr;
    private: uchar *deviceArgMaxMemory = nullptr;

    // Size of input & output for both host & device memories
    private: size_t inputSizeBytes = 1;
    private: size_t outputSizeBytes = 1;
    private: size_t argmaxSizeBytes;

    // brief stream for asyncronization calls for memory copy & execution
    private: cudaStream_t stream;

    private: cv::Size resizeShape;
};
#endif