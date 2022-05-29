#include "BisenetTensorRT.h"

class Logger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

////////////////////////////////////////////////////////////////
BiseNetTensorRT::BiseNetTensorRT(std::string _enginePath)
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

////////////////////////////////////////////////////////////////
BiseNetTensorRT::~BiseNetTensorRT()
{
    cudaFree(this->deviceArgMaxMemory);
    cudaFree(this->deviceInputMemory);
    cudaFree(this->deviceOutputMemory);
    cudaFreeHost(this->hostInputMemory);
    cudaFreeHost(this->hostOutputMemory);
}

////////////////////////////////////////////////////////////////
cv::Mat BiseNetTensorRT::Inference(cv::Mat image)
{
    cudaStreamCreate(&this->stream);
    std::cout << 1 << std::endl;
    // preprocessing transfer the image data to the hostInputMemory
    this->PreProcessing(image);

    std::cout << 2 << std::endl;
    // copy input from host memory to device memroy
    cudaMemcpyAsync(this->deviceInputMemory, this->hostInputMemory,
                    this->inputSizeBytes, cudaMemcpyHostToDevice,
                    this->stream);
    std::cout << 3 << std::endl;

    // TensorRT async execution through cuda context
    // shapres input  (1, 3, 512, 1024)  output  (1, 19, 512, 1024)
    this->context->enqueueV2(this->bindingBuffers.data(), this->stream, nullptr);
    std::cout << 4 << std::endl;

    // Post processing
    // output of argmax is (512, 1024)
    argmaxLaunchKernel(this->deviceOutputMemory, this->deviceArgMaxMemory, stream);
    std::cout << 5 << std::endl;

    cudaMemcpyAsync(this->hostOutputMemory, this->deviceArgMaxMemory, this->argmaxSizeBytes, cudaMemcpyDeviceToHost, stream);
    std::cout << 6 << std::endl;

    cudaStreamSynchronize(stream);
    std::cout << 7 << std::endl;

    // convert output in host memory to cv::Mat
    cv::Mat outputImage(this->resizeShape, CV_8UC1, this->hostOutputMemory);
    return outputImage;
}

////////////////////////////////////////////////////////////////
void BiseNetTensorRT::hwc_to_chw(cv::InputArray src, cv::OutputArray &dst)
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

////////////////////////////////////////////////////////////////
void BiseNetTensorRT::PreProcessing(cv::Mat _image)
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

#if 0
    std::cout << "resize time = " << std::chrono::duration<double>(t2-t1).count() * 1e3 << " ms" << std::endl;
    std::cout << "nchw time = " << std::chrono::duration<double>(t3-t2).count() * 1e3 << " ms" << std::endl;
    std::cout << "copy time = " << std::chrono::duration<double>(t4-t3).count() * 1e3 << " ms" << std::endl;
#endif
}

////////////////////////////////////////////////////////////////
void BiseNetTensorRT::AllocateMemory()
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