#include <bits/stdc++.h>
#include <fstream>
#include <NvInfer.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cudaProfiler.h>
#include <cuda_profiler_api.h>
#include <cuda.h>
#include <chrono>

#include "kernel_launch.h"
#include "visualization.h"

#define INPUT_BINDING_INDEX 0
#define OUTPUT_BINDING_INDEX 1

class Logger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

class BiseNetTensorRT
{
    public: BiseNetTensorRT(std::string _enginePath)
    {
        std::ifstream file(_enginePath, std::ios::binary);
        if(!file)
        {
            std::cout << "error loading engine file [" << _enginePath << "]" << std::endl;
            exit(0);
        }

        file.seekg(0, file.end);
        const int serializedSize = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> engineData(serializedSize);
        char *engineMemory = engineData.data();
        file.read(engineMemory, serializedSize);

        this->runtime = nvinfer1::createInferRuntime(logger);
        this->cudaEngine = runtime->deserializeCudaEngine(engineMemory, serializedSize);

        std::cout << "Loaded cuda engine from " << _enginePath << " at memory " << cudaEngine << std::endl;

        // allocate memory for preprocessed image
        this->resizeShape = cv::Size(1024, 512);

        // allocate memory for input/output bindings
        this->AllocateMemory();

        // create context for execution
        this->context = this->cudaEngine->createExecutionContext();
    }
    public: ~BiseNetTensorRT()
    {
        cudaFree(this->deviceArgMaxMemory);
        cudaFree(this->deviceInputMemory);
        cudaFree(this->deviceOutputMemory);
        cudaFreeHost(this->hostInputMemory);
        cudaFreeHost(this->hostOutputMemory);
    }
    public: cv::Mat Inference(cv::Mat image)
    {
        // preprocessing transfer the image data to the hostInputMemory
        this->PreProcessing(image);

        // copy input from host memory to device memroy
        cudaMemcpyAsync(this->deviceInputMemory, this->hostInputMemory,
                        this->inputSizeBytes, cudaMemcpyHostToDevice,
                        this->stream);

        // TensorRT async execution through cuda context
        // shapres input  (1, 3, 512, 1024)  output  (1, 19, 512, 1024)
        this->context->enqueueV2(this->bindingBuffers.data(), this->stream, nullptr);

        // Post processing
        // output of argmax is (512, 1024)
        argmaxLaunchKernel(this->deviceOutputMemory, this->deviceArgMaxMemory, stream);

        cudaMemcpyAsync(this->hostOutputMemory, this->deviceArgMaxMemory, this->argmaxSizeBytes, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        // convert output in host memory to cv::Mat
        cv::Mat outputImage(this->resizeShape, CV_8UC1, this->hostOutputMemory);
        return outputImage;
    }

    //TensorRT requires your image data to be in NCHW order. But OpenCV reads it in NHWC order.
    void hwc_to_chw(cv::InputArray src, cv::OutputArray &dst)
    {
        std::vector<cv::Mat> channels;
        cv::split(src, channels);

        // Stretch one-channel images to vector
        for (auto &img : channels) {
            std::cout << "before "<< img.size() << " " << img.rows << " " << img.cols << " " << img.channels() << std::endl;
            img = img.reshape(1, 1);
            std::cout << "after "<< img.size() << " " << img.rows << " " << img.cols << " " << img.channels() << std::endl;
        }

        // Concatenate three vectors to one
        cv::hconcat( channels, dst );

        std::cout << "dst "<< dst.size() << " " << dst.rows() << " " << dst.cols() << " " << dst.channels() << std::endl;
    }

    public: void PreProcessing(cv::Mat _image)
    {
        cv::Mat resized, nchw;
        auto t1 = std::chrono::system_clock::now();

        // resizing
        cv::resize(_image, resized, this->resizeShape);


        auto t2 = std::chrono::system_clock::now();

        this->hwc_to_chw(resized, nchw);
        // image = image.astype(float) / 255
        nchw.convertTo(nchw, CV_32FC1, 1.f/255.f);

        auto t3 = std::chrono::system_clock::now();
        memcpy(this->hostInputMemory, (float*)nchw.data, this->inputSizeBytes);
        // cudaMemcpy(this->hostInputMemory, (float*)nchw.data, this->inputSizeBytes, cudaMemcpyHostToHost);
        auto t4 = std::chrono::system_clock::now();

#if 1
        std::cout << "resize time = " << std::chrono::duration<double>(t2-t1).count() * 1e3 << " ms" << std::endl;
        std::cout << "nchw time = " << std::chrono::duration<double>(t3-t2).count() * 1e3 << " ms" << std::endl;
        std::cout << "copy time = " << std::chrono::duration<double>(t4-t3).count() * 1e3 << " ms" << std::endl;
#endif
    }

    private: void AllocateMemory()
    {
        for (int i = 0; i < this->cudaEngine->getNbBindings(); i++)
            std::cout << this->cudaEngine->getBindingName(i) << " , ";
            std::cout << std::endl;

        // allocate page-lock memory(host memory) & GPU device memory for input
        // size of input bindings
        nvinfer1::Dims inputDims = this->cudaEngine->getBindingDimensions(INPUT_BINDING_INDEX);
        size_t inputSizeBytes = 1;
        for (auto d : inputDims.d)
        {
            if (d == 0)
                continue;
            this->inputSizeBytes *= d;
        }
        std::cout << std::endl;
        this->inputSizeBytes *= sizeof(float);

        std::cout << "input size bytes = " << this->inputSizeBytes << " .. count = " << this->inputSizeBytes / sizeof(float) << std::endl;

        // allocate page-lock memory
        cudaHostAlloc((void**)&this->hostInputMemory, this->inputSizeBytes, cudaHostAllocDefault);
        cudaMalloc((void**)&this->deviceInputMemory, this->inputSizeBytes);

        std::cout << "input memories: "<< this->hostInputMemory << " . " << this->deviceInputMemory << std::endl;

        // allocate page-lock memory(host memory) & GPU device memory for output
        nvinfer1::Dims outputDims = this->cudaEngine->getBindingDimensions(OUTPUT_BINDING_INDEX);
        for (auto d : outputDims.d)
        {
            if (d == 0)
                continue;
            this->outputSizeBytes *= d;
        }
        this->outputSizeBytes *= sizeof(float);
        this->argmaxSizeBytes = this->resizeShape.width * this->resizeShape.height * sizeof(uchar);

        std::cout << "output size bytes = " << this->outputSizeBytes 
                  << " .. count = " << this->outputSizeBytes / sizeof(float) 
                  << " .. argmax " << this->argmaxSizeBytes << " and count =" << this->argmaxSizeBytes/sizeof(uchar)
                  << std::endl;

        // allocate page-lock memory & device memory
        cudaHostAlloc((void**)&this->hostOutputMemory, this->argmaxSizeBytes, cudaHostAllocDefault);
        cudaMalloc((void**)&this->deviceOutputMemory, this->outputSizeBytes);
        cudaMalloc((void**)&this->deviceArgMaxMemory, this->argmaxSizeBytes);

        std::cout << "output memories: " << this->hostOutputMemory << " . " << this->deviceOutputMemory << std::endl;

        // Bindings buffers of device for Cuda context execution
        this->bindingBuffers.push_back(this->deviceInputMemory);
        this->bindingBuffers.push_back(this->deviceOutputMemory);

        // Stream for syncronization
        cudaStreamCreate(&this->stream);
    }

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

cv::Mat global_image;
void mouseHandler(int event,int x,int y, int flags,void* param)
{
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
        std::cout << semantic.size() << " , channels " << semantic.channels() << std::endl;
        std::cout << coloredSemantic.size() << " , channels " << coloredSemantic.channels() << std::endl;

        // just for visualization ,add 20 to brightness the image, as ids is [0-19] which is really dark
        semantic.convertTo(semantic,CV_8UC1, 1, 20);

        cv::imshow("image", image);
        cv::imshow("semantic", semantic);
        std::cout << "colored semantic type = " << coloredSemantic.type() << " " << image.type() <<  std::endl;
        cv::imshow("coloredSemantic", coloredSemantic);

        global_image = semantic;
        cv::setMouseCallback("semantic", mouseHandler, &semantic);
        
        if (cv::waitKey(0) == 27)
        {
            cv::destroyAllWindows();
            return 0;
        }

    }
    return 0;
}