#include <bits/stdc++.h>
#include <NvInfer.h>
#include <fstream>


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
        this->enginePath = _enginePath;
        std::ifstream file(this->enginePath, std::ios::binary);
        if(!file)
        {
            std::cout << "error loading engine file [" << this->enginePath << "]" << std::endl;
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

        std::cout << "Loaded cuda engine from " << this->enginePath << " at memory " << cudaEngine << std::endl;
    }    
    public: ~BiseNetTensorRT()
    {
        delete this->cudaEngine;
        delete this->runtime;
    }
    public: void PreProcessing()
    {

    }

    public: void PostProcessing()
    {

    }

    private: nvinfer1::IRuntime *runtime;
    private: nvinfer1::ICudaEngine *cudaEngine;
    private: std::string enginePath;
};


int main(int argc, char** argv)
{
    std::string enginePath = "/home/amrelsersy/PointPainting/bisenet_tensorrt.trt";

    if (argc > 1)
        enginePath = argv[1];

    auto bisenet = BiseNetTensorRT(enginePath);   
    return 0;
}