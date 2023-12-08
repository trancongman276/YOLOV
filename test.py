import argparse
import os
from loguru import logger
import cv2
import numpy as np

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="whether the input shape should be dynamic or not",
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--decode_in_inference", action="store_true", help="decode in inference or not"
    )

    return parser


args = make_parser().parse_args("")
args.exp_file = "./exps/yolov/yolov_l.py"
args.name = "yolov_l"
args.ckpt = "./yolov_l.pth"
args.output_name = "yolov_l.onnx"


logger.info("args value: {}".format(args))
exp = get_exp(args.exp_file, args.name)
exp.merge(args.opts)

if not args.experiment_name:
    args.experiment_name = exp.exp_name

model = exp.get_model()
ckpt_file = args.ckpt

# load the model state dict
ckpt = torch.load(ckpt_file, map_location="cuda")

model.eval()
if "model" in ckpt:
    ckpt = ckpt["model"]
model.load_state_dict(ckpt)
model = replace_module(model, nn.SiLU, SiLU)
model.head.decode_in_inference = args.decode_in_inference
model.to("cuda")
logger.info("loading checkpoint done.")

image = cv2.imread("./assets/dog.jpg")
image = cv2.resize(image, (512, 512))
images = np.array([image, image, image, image])
images = torch.from_numpy(images).float().cuda()
images = images.permute(0, 3, 1, 2)
test = model(
    images,
)
