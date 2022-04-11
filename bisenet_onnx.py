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


def export_onnx(args):
    # Load Weights of Pytorch model
    bisenetv2 = BiSeNetV2()
    checkpoint = torch.load(args.weights_path, map_location=dev)
    bisenetv2.load_state_dict(checkpoint['bisenetv2'], strict=False)
    bisenetv2.eval()
    bisenetv2.to(device)

    # Dummy input for onnx tracing
    image = cv2.imread(args.image_path)
    image_input = preprocessing_kitti(image)
    image_input.requires_grad = True
    print("input image ", image.shape, image_input.shape) # torch.Size([1, 3, 512, 1024])

    torch.cuda.empty_cache() 
    # Export the model
    torch.onnx.export(bisenetv2, image_input, args.onnx_path, export_params=True,  opset_version=10, do_constant_folding=True,
                        input_names = ['input'], output_names = ['output'],
                        # dynamic_axes={
                        #     'input':{0:'batch_size'},
                        #     'output':{0:'batch_size'}
                        # }
                        )

    print("Export BiseNetv2 model to onnx format in ", args.onnx_path)

    # Check valid graph of onnx 
    # print(check_onnx_valid_structure(args))

def onnx_runtime_test():
    image = cv2.imread(args.image_path)
    image_input = preprocessing_kitti(image)
    image_input.requires_grad = True

    # runtime_session
    ort_session = onnxruntime.InferenceSession(args.onnx_path)
    # print(ort_session.get_inputs()[0], ort_session.get_outputs()[0])

    ort_input = image_input.detach().cpu().numpy()
    # print(ort_input.shape)
    ort_input_dict = {
        "input": ort_input
    }

    # inference
    t1 = time.time()
    ort_outputs = ort_session.run(None, input_feed=ort_input_dict)
    ort_output_image = ort_outputs[0]
    t2 = time.time()
    print('onnxruntime inference ', (t2-t1)*1000, ' ms')
    # postprocessing in torch cpu
    semantic = postprocessing(torch.from_numpy(ort_output_image))

    # visualization
    visualizer = Visualizer('2d')
    semantic = visualizer.get_colored_image(image, semantic)
    print(semantic.shape)
    
    cv2.imshow('ort_output', semantic)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()

def check_onnx_valid_structure(args):
    onnx_model = onnx.load(args.onnx_path)
    return onnx.checker.check_model(onnx_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='Kitti_sample/image_2/000038.png')
    parser.add_argument('--weights_path', type=str, default='BiSeNetv2/checkpoints/BiseNetv2_150.pth')
    parser.add_argument('--onnx_path', type=str, default='bisenet.onnx')
    parser.add_argument('--ort', action='store_true')
    args = parser.parse_args()

    if args.ort:
        onnx_runtime_test()
    else:
        export_onnx(args)
