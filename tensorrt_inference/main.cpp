#include <bits/stdc++.h>
#include <fstream>
#include <NvInfer.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

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

        this->context = this->cudaEngine->createExecutionContext();
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
        // preprocessing
        image = this->PreProcessing(image);

        // flatten cv::Mat to array
        float *flattenImage = (float *)image.data;

        // copy the flatten image to page-lock(host) input memory
        memcpy((float *)this->hostInputMemory, flattenImage, this->inputSizeBytes);

        // copy input from host memory to device memroy
        cudaMemcpyAsync(this->deviceInputMemory, this->hostInputMemory, 
                        this->inputSizeBytes, cudaMemcpyHostToDevice, 
                        *this->stream);

        // TensorRT async execution through cuda context
        this->context->enqueueV2(this->bindingBuffers.data(), *this->stream, nullptr);

        // copy output from device memory to host memory
        cudaMemcpyAsync(this->hostOutputMemory, this->deviceOutputMemory,
                        this->outputSizeBytes, cudaMemcpyDeviceToHost, 
                        *this->stream);

        cudaStreamSynchronize(*stream);
        
        // convert output in host memory to cv::Mat
        // cv::Mat outputImage;
        // outputImage.data = new uchar(this->outputSizeBytes);
        // memcpy(outputImage.data, (uchar *)this->hostOutputMemory, this->outputSizeBytes);

        // Post processing

        // visualization
    }
    public: cv::Mat PreProcessing(cv::Mat _image)
    {
        cv::Mat resized;
        cv::resize(_image, resized, cv::Size(1024, 512));

        cv::Mat flattenImage;
        resized.convertTo(flattenImage, CV_32FC3, 1.f/255.f);
        return flattenImage;

        /* We may change the order of dims in the input array, it must be at some format ex: NHWC
            std::vector< cv::cuda::GpuMat > chw;
                for (size_t i = 0; i < channels; ++i)
                {
                    chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
                }
                cv::cuda::split(flt_image, chw);
            }
        */
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

        std::cout << "before input memories: "<< this->hostInputMemory << " . " << this->deviceInputMemory << std::endl;

        // allocate page-lock memory
        cudaHostAlloc(this->hostInputMemory, this->inputSizeBytes, cudaHostAllocDefault);
        cudaMalloc(this->deviceInputMemory, this->inputSizeBytes);

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
        cudaHostAlloc(this->hostOutputMemory, this->outputSizeBytes, cudaHostAllocDefault);
        cudaMalloc(this->deviceInputMemory, this->outputSizeBytes);

        std::cout << "output memories: " << this->hostInputMemory << " . " << this->deviceInputMemory << std::endl;

        // Bindings buffers of device for Cuda context execution
        this->bindingBuffers.push_back(*this->deviceInputMemory);
        this->bindingBuffers.push_back(*this->deviceOutputMemory);

        // Stream for syncronization
        cudaStreamCreate(this->stream);
    }

    private: nvinfer1::IRuntime *runtime;
    private: nvinfer1::ICudaEngine *cudaEngine;
    private: nvinfer1::IExecutionContext *context;

    // Bindings buffers of device for Cuda context execution
    private: std::vector<void *> bindingBuffers;

    // Page-lock and GPU device memory for input & output
    private: void **hostInputMemory = nullptr;
    private: void **deviceInputMemory = nullptr;
    private: void **hostOutputMemory = nullptr;
    private: void **deviceOutputMemory = nullptr;

    // Size of input & output for both host & device memories
    private: size_t inputSizeBytes = 1;
    private: size_t outputSizeBytes = 1;

    // brief stream for asyncronization calls for memory copy & execution
    private: cudaStream_t *stream;
};


int main(int argc, char** argv)
{
    std::string enginePath = "/home/amrelsersy/PointPainting/bisenet_tensorrt.trt";
    if (argc > 1)
        enginePath = argv[1];

    auto bisenet = BiseNetTensorRT(enginePath);   

    std::string rootPath = "/home/amrelsersy/PointPainting/Kitti_sample/";
    for (const auto & entry : std::filesystem::directory_iterator(rootPath))
    {   
        std::string imagePath = entry.path().string();
        cv::Mat image = cv::imread(imagePath, cv::ImreadModes::IMREAD_COLOR);
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