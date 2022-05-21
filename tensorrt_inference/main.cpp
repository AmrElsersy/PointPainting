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
#include "argmax_cuda.h"

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

        // allocate memory for input/output bindings
        this->AllocateMemory();
        // create context for execution
        this->context = this->cudaEngine->createExecutionContext();

        // allocate memory for preprocessed image
        this->resizeShape = cv::Size(1024, 512);
        this->NCHW_preprocessed_buffer.resize(std::size_t(resizeShape.width * resizeShape.height));
    }
    public: ~BiseNetTensorRT()
    {
        delete this->cudaEngine;
        delete this->runtime;

        cudaFree(this->deviceInputMemory);
        cudaFree(this->deviceOutputMemory);
        cudaFreeHost(this->hostInputMemory);
        cudaFreeHost(this->hostOutputMemory);
    }
    public: void Inference(cv::Mat image)
    {
        // preprocessing transfer the image data to the hostInputMemory
        this->PreProcessing(image);

        // copy input from host memory to device memroy
        cudaMemcpyAsync(this->deviceInputMemory, this->hostInputMemory,
                        this->inputSizeBytes, cudaMemcpyHostToDevice,
                        *this->stream);

        for (auto b: bindingBuffers)
            std::cout << b << " ";
            std::cout << std::endl;

        // TensorRT async execution through cuda context
        this->context->enqueueV2(this->bindingBuffers.data(), *this->stream, nullptr);

        // shapres input  (1, 3, 512, 1024)  output  (1, 19, 512, 1024)
        cudaStreamSynchronize(*stream);

        // Post processing
        // output of argmax is (512, 1024)
        argmax(this->deviceOutputMemory, this->deviceArgMaxMemory);

        // copy output from device memory to host memory
        // cudaMemcpyAsync(this->hostOutputMemory, this->deviceOutputMemory,
        //                 this->outputSizeBytes, cudaMemcpyDeviceToHost,
        //                 *this->stream);
        cudaMemcpy(this->hostOutputMemory, this->deviceArgMaxMemory, this->outputSizeBytes, cudaMemcpyDeviceToHost);

        // convert output in host memory to cv::Mat
        cv::Mat outputImage(this->resizeShape, CV_32FC3, this->hostOutputMemory);

        // visualization
        cv::imshow("outputImage", outputImage);
        cv::waitKey(0);
    }

    void hwc_to_chw(cv::InputArray src, cv::OutputArray &dst)
    {
        std::vector<cv::Mat> channels;
        cv::split(src, channels);

        // Stretch one-channel images to vector
        for (auto &img : channels) {
            img = img.reshape(1, 1);
        }

        // Concatenate three vectors to one
        cv::hconcat( channels, dst );
    }

    public: void PreProcessing(cv::Mat _image)
    {
        cv::Mat resized;
        auto t1 = std::chrono::system_clock::now();

        cv::resize(_image, resized, this->resizeShape);

        auto t2 = std::chrono::system_clock::now();
        // resized.convertTo(resized, CV_32FC3, 1.f/255.f);


        //TensorRT requires your image data to be in NCHW order. But OpenCV reads it in NHWC order.
        // std::vector<cv::Mat> chw;
        // auto t3 = std::chrono::system_clock::now();
        // for (size_t n = 0; n < 3; ++n)
        //     chw.emplace_back(cv::Mat(this->resizeShape, CV_32FC1,
        //         this->NCHW_preprocessed_buffer.data() + n * this->resizeShape.width * this->resizeShape.height));
        // auto t4 = std::chrono::system_clock::now();
        // cv::split(resized, chw);
        // auto t5 = std::chrono::system_clock::now();

        auto width = this->resizeShape.width;
        auto height = this->resizeShape.height;
        auto size = height * width;

        for (unsigned j = 0, volChl = height * width; j < height; ++j)
        {
            for( unsigned k = 0; k < width; ++ k)
            {
                cv::Vec3b bgr = resized.at<cv::Vec3b>(j,k);
                hostInputMemory[0 * volChl + j * width + k] = (2.0 / 255.0) * float(bgr[2]) - 1.0;
                hostInputMemory[1 * volChl + j * width + k] = (2.0 / 255.0) * float(bgr[1]) - 1.0;
                hostInputMemory[2 * volChl + j * width + k] = (2.0 / 255.0) * float(bgr[0]) - 1.0;
            }
        }

        auto t3 = std::chrono::system_clock::now();

#if 1
        std::cout << "resize time = " << std::chrono::duration<double>(t2-t1).count() * 1e3 << " ms" << std::endl;
        std::cout << "/255 time = " << std::chrono::duration<double>(t3-t2).count() * 1e3 << " ms" << std::endl;
        // std::cout << "chw time = " << std::chrono::duration<double>(t4-t3).count() * 1e3 << " ms" << std::endl;
        // std::cout << "split time = " << std::chrono::duration<double>(t5-t4).count() * 1e3 << " ms" << std::endl;
        // std::cout << "total time = " << std::chrono::duration<double>(t5-t1).count() * 1e3 << " ms" << std::endl;
#endif
    }

    public: void PostProcessing()
    {

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

        std::cout << "output size bytes = " << this->outputSizeBytes << " .. count = " << this->outputSizeBytes / sizeof(float) << std::endl;

        // allocate page-lock memory & device memory
        cudaHostAlloc((void**)&this->hostOutputMemory, this->outputSizeBytes, cudaHostAllocDefault);
        cudaMalloc((void**)&this->deviceOutputMemory, this->outputSizeBytes);

        this->argmaxSizeBytes = this->resizeShape.width * this->resizeShape.height * sizeof(float);
        cudaMalloc((void**)&this->deviceArgMaxMemory, this->argmaxSizeBytes);

        std::cout << "output memories: " << this->hostOutputMemory << " . " << this->deviceOutputMemory << std::endl;

        // Bindings buffers of device for Cuda context execution
        this->bindingBuffers.push_back(this->deviceInputMemory);
        this->bindingBuffers.push_back(this->deviceOutputMemory);

        // Stream for syncronization
        cudaStreamCreate(this->stream);
    }

    private: nvinfer1::IRuntime *runtime;
    private: nvinfer1::ICudaEngine *cudaEngine;
    private: nvinfer1::IExecutionContext *context;

    // Bindings buffers of device for Cuda context execution
    private: std::vector<void *> bindingBuffers;

    // Page-lock and GPU device memory for input & output
    private: float *hostInputMemory = nullptr;
    private: float *deviceInputMemory = nullptr;
    private: float *hostOutputMemory = nullptr;
    private: float *deviceOutputMemory = nullptr;
    private: float *deviceArgMaxMemory = nullptr;

    // Size of input & output for both host & device memories
    private: size_t inputSizeBytes = 1;
    private: size_t outputSizeBytes = 1;
    private: size_t argmaxSizeBytes;

    // brief stream for asyncronization calls for memory copy & execution
    private: cudaStream_t *stream;

    private: cv::Size resizeShape;
    private: std::vector<float> NCHW_preprocessed_buffer;
};


int main(int argc, char** argv)
{
    std::string enginePath = "/home/amrelsersy/PointPainting/bisenet_tensorrt.trt";
    if (argc > 1)
        enginePath = argv[1];

    auto bisenet = BiseNetTensorRT(enginePath);

    std::string rootPath = "/home/amrelsersy/PointPainting/Kitti_sample/image_2";
    for (const auto & entry : std::filesystem::directory_iterator(rootPath))
    {
        std::string imagePath = entry.path().string();
        cv::Mat image = cv::imread(imagePath, cv::ImreadModes::IMREAD_COLOR);
        std::cout << imagePath << " " << image.size() << std::endl;
        cv::imshow("image", image);
        if (cv::waitKey(0) == 27)
        {
            cv::destroyAllWindows();
            return 0;
        }

        bisenet.Inference(image);
    }
    return 0;
}