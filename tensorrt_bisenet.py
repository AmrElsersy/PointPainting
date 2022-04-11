import onnx
import onnxruntime
import torch
import cv2
import numpy as np
import argparse, time
from BiSeNetv2.model.BiseNetv2 import BiSeNetV2
from BiSeNetv2.utils.utils import preprocessing_kitti, postprocessing
from visualizer import Visualizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

INPUT_BINDING = 0
OUTPUT_BINDING = 1

def build_engine(args):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # builder to build the cuda engine from network definition
    builder = trt.Builder(TRT_LOGGER)

    # allocate memory for network definition
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network_definition = builder.create_network(explicit_batch)

    # parse onnx file to a network definition
    onnx_parser = trt.OnnxParser(network_definition, TRT_LOGGER)
    print(builder, network_definition, onnx_parser)
    parsed = onnx_parser.parse_from_file(args.onnx_path)

    if not parsed:
        print("Error in parsing onnx file model")
        exit()
    print("Network is built through onnx parser from ", args.onnx_path)
    
    
    # used in config to determine the dynamic shape inputs
    # optim_profile = builder.create_optimization_profile()
    # optim_profile.set_shape("batch_size", min=(1, 3, 512, 1024), opt=(1, 3, 512, 1024), max=(10, 3, 512, 1024))
    # print(optim_profile.get_shape('batch_size'))

    # build engine creation configuration
    config = builder.create_builder_config()
    # config.flags =  1 << int(trt.BuilderFlag.FP16) # persesion FP16
    config.max_workspace_size = 1 << 28 # 256MiB size used for cuda engine
    # config.add_optimization_profile(optim_profile)
    
    # build cuda engine
    engine = builder.build_engine(network_definition, config)

    # another way to build cuda engine, serialize the network into host memory buffer then deserialize it from the buffer to cuda engine
    # serialize means converting to a buffer memory that can be saved, so that user will not wait till the engine is build at the begin of the program
    # each time, so it serialize once and deserializes it every run
    # plan_host_memory = builder.build_serialized_network(network_definition, config)
    # trt_runtime = trt.Runtime(TRT_LOGGER)
    # cuda_engine = trt_runtime.deserialize_cuda_engine(plan_host_memory)
    # print(engine, config, plan_host_memory, trt_runtime, cuda_engine)

    return engine

def tensorrt_inference(args):
    # input
    image = cv2.imread(args.image_path)
    image_input = preprocessing_kitti(image)

    # tensorrt cuda engine for inference
    cuda_engine = build_engine(args)
    
    # create context(session) for optimized inference
    context = cuda_engine.create_execution_context()

    t1 = time.time()
    # =========== create memory for input host(cpu) and device(gpu) =========== 
    size = trt.volume(cuda_engine.get_binding_shape(INPUT_BINDING))
    dtype = trt.nptype(cuda_engine.get_binding_dtype(INPUT_BINDING))
    input_host_memory = cuda.pagelocked_empty(size, dtype)
    input_device_memory = cuda.mem_alloc(input_host_memory.nbytes)
    t2 = time.time()
    #  =========== create memory for OUTPUT host(cpu) and device(gpu) =========== 
    size = trt.volume(cuda_engine.get_binding_shape(OUTPUT_BINDING))
    dtype = trt.nptype(cuda_engine.get_binding_dtype(OUTPUT_BINDING))
    output_host_memory = cuda.pagelocked_empty(size, dtype)
    output_device_memory = cuda.mem_alloc(output_host_memory.nbytes)
    t3 = time.time()

    bindings = [int(input_device_memory), int(output_device_memory)]
    stream = cuda.Stream()
    
    # Copy our image to the pagelocked input host buffer
    # print(image_input.shape)
    image_input_flatten = image_input.detach().cpu().numpy().ravel()
    # print(image_input_flatten.shape)
    np.copyto(input_host_memory, image_input_flatten)
    t4 = time.time()

    # transfer input host memory to device memory
    cuda.memcpy_htod_async(input_device_memory, input_host_memory, stream)
    t5 = time.time()
    # tensorrt runtime execution
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    t6 = time.time()
    # transfer output device memory to host memory
    cuda.memcpy_dtoh_async(output_host_memory, output_device_memory, stream)

    t7 = time.time()
    # synchronize
    stream.synchronize()

    output = output_host_memory.reshape(cuda_engine.get_binding_shape(OUTPUT_BINDING))
    t8 = time.time()

    # postprocessing and visualization
    semantic = postprocessing(torch.from_numpy(output))
    t9 = time.time()

    print("allocate input ", (t2-t1)*1000, 'ms')
    print("allocate output ", (t3-t2)*1000, 'ms')
    print("copy our img ", (t4-t3)*1000, 'ms')
    print("host to device input ", (t5-t4)*1000, 'ms')
    print("inference ", (t6-t5)*1000, 'ms')
    print("device to host output", (t7-t6)*1000, 'ms')
    print("reshape output ", (t8-t7)*1000, 'ms')
    print("postprocessing ", (t9-t8)*1000, 'ms')
    print("total ", (t9-t3)*1000, 'ms')

    visualizer = Visualizer('2d')
    semantic = visualizer.get_colored_image(image, semantic)
    print(semantic.shape)    
    cv2.imshow('ort_output', semantic)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='Kitti_sample/image_2/000038.png')
    parser.add_argument('--onnx_path', type=str, default='bisenet.onnx')
    args = parser.parse_args()

    tensorrt_inference(args)