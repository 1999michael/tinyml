import torch
import argparse
from models import *
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('--resolution', default=112, type=int)
parser.add_argument('--onnyx_op_ver', default=14, type=int)
parser.add_argument('--arch', default='resnet20', type=str)
parser.add_argument('--layers', default=2, type=int)
parser.add_argument('--dir', required=True, type=str)

def torch_to_onnx(args):
    sample_input = torch.rand((1, 3, args.resolution, args.resolution))
    model = models_dict[args.arch](num_layers=args.layers)
    checkpoint = torch.load(args.dir + "\\model.tar", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    onnx_model_path = args.dir + "\\onnyx_model_" + str(args.onnyx_op_ver) + ".onnx"
    torch.onnx.export(
        model,                  # PyTorch Model
        sample_input,           # Input tensor
        onnx_model_path,        # Output file (eg. 'output_model.onnx')
        opset_version=12,       # Operator support version
        input_names=['input'],  # Input tensor name (arbitary)
        output_names=['output'] # Output tensor name (arbitary)
    )
    return onnx_model_path

def onnx_to_tf(args, onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_model_path = args.dir + "\\tf_model_" + str(args.onnyx_op_ver)
    tf_rep.export_graph(tf_model_path)
    return tf_model_path

def tf_to_tflite(args, tf_model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.target_spec.supported_types = [tf.uint8]
    tflite_model = converter.convert()

    # Save the model
    tflite_model_path = args.dir + "\\tflite_" + str(args.onnyx_op_ver) + ".lite"
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

def main(args):
    onnx_model_path = torch_to_onnx(args)
    tf_model_path = onnx_to_tf(args, onnx_model_path)
    tf_to_tflite(args, tf_model_path)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
