#include <NvInfer.h>
#include <bits/stdc++.h>
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"
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

int main(int argc, char** argv)
{
    std::string onnxPath = "/home/amrelsersy/PointPainting/bisenet.onnx";
    std::string enginePath = "/home/amrelsersy/PointPainting/";

    if (argc > 2)
    {
        // first argument is onnx path and second is the saved engine path
        onnxPath = argv[1];
        enginePath = argv[2];
    }

    // Builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

    // Configuration
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20);
    // if (builder->platformHasFastFp16())
    //     config->setFlag(nvinfer1::BuilderFlag::kFP16);
    builder->setMaxBatchSize(1); // 1 image inference        

    // Network
    nvinfer1::NetworkDefinitionCreationFlags flags = 1U <<  static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(flags);

    // ONNX Parser
    nvonnxparser::IParser *onnxParser = nvonnxparser::createParser(*network, logger);
    if (!onnxParser->parseFromFile(onnxPath.c_str(), (int)nvinfer1::ILogger::Severity::kINFO))
    {
        std::cout << "Error Parsing ONNX file" << std::endl;
        exit(0);
    }

    std::cout << "Loaded ONNX file" << std::endl;

    nvinfer1::IHostMemory *serializedEngine = builder->buildSerializedNetwork(*network, *config);

    if (enginePath[enginePath.size()-1] == '/')
        enginePath += "bisenet_tensorrt.trt";
    else
        enginePath += "/bisenet_tensorrt.trt";

    std::ofstream file(enginePath, std::ios::binary | std::ios::out);
    if (!file)
    {
        std::cout << "Error cannot save engine in path: " << enginePath << std::endl;
        exit(0);
    }
    file.write((char*)serializedEngine->data(), serializedEngine->size());
    file.close();

    std::cout << "Cuda Engine of size" << serializedEngine->size() << " saved at path: " << enginePath << std::endl;
    return 0;
}