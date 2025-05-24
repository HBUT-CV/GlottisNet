import argparse
import os.path as osp

import numpy as np
import onnx
import onnxruntime as rt
import torch

from mmdet.apis import init_detector


def pytorch2onnx(config_path,
                 checkpoint_path,
                 device,
                 Input,
                 opset_version=11,
                 output_file='tmp.onnx'):

    # prepare original model for converting onnx model
    model = init_detector(config_path, checkpoint_path, device=device)
    # model.eval()

    torch.onnx.export(
        model,
        Input,
        output_file,
        opset_version=opset_version)

    print(f'Successfully exported ONNX model: {output_file}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[800, 1216],
        help='input image size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

    Input = torch.randn(1, 3, args.shape[0], args.shape[1]).to(args.device)

    # convert model to onnx file
    pytorch2onnx(
        args.config,
        args.checkpoint,
        args.device,
        Input,
        opset_version=args.opset_version,
        output_file=args.output_file)
