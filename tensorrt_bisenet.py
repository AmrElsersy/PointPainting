import onnx
import onnxruntime
import torch
import cv2
import numpy as np
import argparse, time, os
from BiSeNetv2.model.BiseNetv2 import BiSeNetV2
from visualizer import Visualizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

INPUT_BINDING = 0
OUTPUT_BINDING = 1
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

import torchvision.transforms.transforms as T

class TensorRT_Bisenet:
    def __init__(self, args):
        self.args = args
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        if os.path.exists(args.engine_path):
            print("loading cuda engine from [", args.engine_path,'] ...')
            # load the cuda engine
            with open(args.engine_path, 'rb') as f:
                serialized_engine = f.read()
            # tensorrt runtime to load the engine
            trt_runtime = trt.Runtime(self.TRT_LOGGER)
            self.cuda_engine = trt_runtime.deserialize_cuda_engine(serialized_engine)
            print('load cuda engine: ', self.cuda_engine)

        else:
            self.cuda_engine = self.serialize_cuda_engine()
            print('saved cuda engine in [', args.engine_path, '].')

        self.tensorrt_allocate_memory()

    def serialize_cuda_engine(self):
        # builder to build the cuda engine from network definition
        builder = trt.Builder(self.TRT_LOGGER)

        # allocate memory for network definition
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network_definition = builder.create_network(explicit_batch)

        # parse onnx file to a network definition
        onnx_parser = trt.OnnxParser(network_definition, self.TRT_LOGGER)
        print(builder, network_definition, onnx_parser)
        parsed = onnx_parser.parse_from_file(self.args.onnx_path)

        if not parsed:
            print("Error in parsing onnx file model")
            exit()
        print("Network is built through onnx parser from ", self.args.onnx_path)
        
        
        # used in config to determine the dynamic shape inputs
        # optim_profile = builder.create_optimization_profile()
        # optim_profile.set_shape("batch_size", min=(1, 3, 512, 1024), opt=(1, 3, 512, 1024), max=(10, 3, 512, 1024))
        # print(optim_profile.get_shape('batch_size'))

        # build engine creation configuration
        config = builder.create_builder_config()
        # config.flags =  1 << int(trt.BuilderFlag.FP16) # persesion FP16
        config.max_workspace_size = 1 << 28 # 256MiB size used for cuda engine
        # config.add_optimization_profile(optim_profile)
        
        serialized_engine_host_memory = builder.build_serialized_network(network_definition, config)
        # save the serialized engine and load it later using trt runtime
        with open(self.args.engine_path, 'wb') as f:
            f.write(serialized_engine_host_memory)

        # build cuda engine directly to return it
        engine = builder.build_engine(network_definition, config)
        return engine

    def test_data(self):
        visualizer = Visualizer('2d')

        dataset_path = args.data_path
        images_paths = os.listdir(dataset_path)
        images_paths = [os.path.join(dataset_path, name) for name in images_paths]
        for path in images_paths:
            image = cv2.imread(path)
            # tensorrt
            semantic = self.tensorrt_inference(image)
            # visualization
            semantic = visualizer.get_colored_image(image, semantic)
            # print(semantic.shape)    
            cv2.imshow('ort_output', semantic)
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                return

    def tensorrt_allocate_memory(self):   
        # create context(session) for optimized inference
        self.context = self.cuda_engine.create_execution_context()
        t1 = time.time()
        # =========== create memory for input host(cpu) and device(gpu) =========== 
        size = trt.volume(self.cuda_engine.get_binding_shape(INPUT_BINDING))
        dtype = trt.nptype(self.cuda_engine.get_binding_dtype(INPUT_BINDING))
        self.input_host_memory = cuda.pagelocked_empty(size, dtype)
        self.input_device_memory = cuda.mem_alloc(self.input_host_memory.nbytes)
        t2 = time.time()
        #  =========== create memory for OUTPUT host(cpu) and device(gpu) =========== 
        size = trt.volume(self.cuda_engine.get_binding_shape(OUTPUT_BINDING))
        dtype = trt.nptype(self.cuda_engine.get_binding_dtype(OUTPUT_BINDING))
        self.output_host_memory = cuda.pagelocked_empty(size, dtype)
        self.output_device_memory = cuda.mem_alloc(self.output_host_memory.nbytes)
        t3 = time.time()
        # connected bindings
        self.bindings = [int(self.input_device_memory), int(self.output_device_memory)]
        # stream to syncrnoize async calls of copying memory and kernel execution
        self.stream = cuda.Stream()
        
    def tensorrt_inference(self, image):
        # input
        t1 = time.time()
        image_input = self.preprocessing_numpy(image)
        
        t2 = time.time()
        # Flatten the image to put it in a sequential array
        image_input_flatten = image_input.ravel()
        t3 = time.time()
        # Copy our image to the page-locked input host buffer
        np.copyto(self.input_host_memory, image_input_flatten)
        t4 = time.time()

        # transfer input host memory to device memory
        cuda.memcpy_htod_async(self.input_device_memory, self.input_host_memory, self.stream)
        t5 = time.time()
        # tensorrt runtime execution
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        t6 = time.time()
        # transfer output device memory to host memory
        cuda.memcpy_dtoh_async(self.output_host_memory, self.output_device_memory, self.stream)

        t7 = time.time()
        # synchronize waits until all preceding commands in the given stream have completed
        # https://leimao.github.io/blog/CUDA-Stream/ for asyncronization execution
        # https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf 
        self.stream.synchronize()
        t8 = time.time()

        output = self.output_host_memory.reshape(self.cuda_engine.get_binding_shape(OUTPUT_BINDING))
        t9 = time.time()

        # postprocessing and visualization
        semantic = self.postprocessing_numpy(output.squeeze(0))

        print("preprocessing ", (t2-t1)*1000, 'ms')
        print("flatten ", (t3-t2)*1000, 'ms')
        print("copy to host ", (t4-t3)*1000, 'ms')
        print("host to device input ", (t5-t4)*1000, 'ms')
        print("inference ", (t6-t5)*1000, 'ms')
        print("device to host output", (t7-t6)*1000, 'ms')
        print("syncronize ", (t8-t7)*1000, 'ms')       
        print("reshape output ", (t9-t8)*1000, 'ms')
        print("total ", (t9-t3)*1000, 'ms')

        return semantic

    def preprocessing_numpy(self, image):
        new_shape = (1024,512)
        image = cv2.resize(image, new_shape)
        image = T.ToTensor()(image).unsqueeze(0)
        return image.numpy()

    def postprocessing_numpy(self, pred):
        semantic = pred.argmax(axis=0)
        return semantic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='Kitti_sample/image_2/000038.png')
    parser.add_argument('--onnx_path', type=str, default='bisenet.onnx')
    parser.add_argument('--engine_path', type=str, default='serialized_cuda_engine.trt')
    parser.add_argument('--data_path', type=str, default='data/KITTI/testing/image_2')
    args = parser.parse_args()

    fast_bisenet = TensorRT_Bisenet(args)
    fast_bisenet.test_data()