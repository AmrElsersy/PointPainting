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

    ///\brief Do inference of Bisenetv2 TensorRT model on the image to generate semantic semgnetation map
    ///\param[in] image input image from opencv of type CV_8UC3
    ///\return opencv image contains semantic segmenation map of type CV_8UC1, each pixel contains 8-bit semantic id value
    public: cv::Mat Inference(cv::Mat image);
    
    ///\brief convert the input image from NHWC format to NCHW format 
    /// as TensorRT requires your image data to be in NCHW order. But OpenCV reads it in NHWC order.
    ///\param[in] src input image in NHWC format
    ///\param[in] dst output image in NCHW format
    void hwc_to_chw(cv::InputArray src, cv::OutputArray &dst);

    ///\brief do preprocessing(resizing, normalizing, converting to nchw) on the input image and store it in the page-locked input memory
    ///\param[in] image input opencv image of type CV_8UC3
    public: void PreProcessing(cv::Mat _image);

    ///\brief Allocate GPU Memory & CPU Page-Locked memory for inputs/outputs
    private: void AllocateMemory();

    ///\brief runtime to create the optimized cuda engine
    private: nvinfer1::IRuntime *runtime;

    ///\brief cuda engine for bisenetv2 model from its onnx format
    private: nvinfer1::ICudaEngine *cudaEngine;

    ///\brief execution context for inference
    private: nvinfer1::IExecutionContext *context;

    ///\brief Bindings buffers of device for Cuda context execution
    private: std::vector<void *> bindingBuffers;

    ///\brief Page-lock and GPU device memory for input & output
    private: float *hostInputMemory = nullptr;
    private: float *deviceInputMemory = nullptr;
    private: float *deviceOutputMemory = nullptr;
    private: uchar *hostOutputMemory = nullptr;
    private: uchar *deviceArgMaxMemory = nullptr;

    ///\brief Size of input & output for both host & device memories
    private: size_t inputSizeBytes = 1;
    private: size_t outputSizeBytes = 1;
    private: size_t argmaxSizeBytes;

    ///\brief stream for asyncronization calls for memory copy & execution
    private: cudaStream_t stream;

    ///\brief preprocessing resize shape of the input image
    private: cv::Size resizeShape;
};
#endif