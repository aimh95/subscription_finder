from utils.test_dataset_gen import get_input_data
from utils.utilities import crop_img, numpyIMG_resize, cc_map_postprocessing, min_max_norm, sobel_operation
from models.real_u_net import U_Net
from utils.test_dataset_gen import unsqueeze
import matplotlib.pyplot as plt
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from data_gen import data_generation_test_ver

padding = 100



model = U_Net()

video_path = "./datasets/test_dataset/yesman.mp4" #dataset_path
output_video_dir = "./results/"
# model_weight_path = "weight_path/real_u_net_light_totaltext_150epochs_load/cp-0141.ckpt"
model_weight_path = "weight_path/real_u_net_totaltext_100epochs/cp-0071.ckpt"

test_data_loader = data_generation_test_ver.ReadVideoAsData(dataset_path=video_path, output_path=output_video_dir, model_weight_path=model_weight_path, model=model)

test_data_loader._run()

#ndarray: 232 880 3 [[[0.17647059 0.12156863 0.06666667],  [0.17647059 0.12156863 0.06666667],  [0.17647059 0.12156863 0.06666667],  ...,  [0.13333333 0.08627451 0.0745098 ],  [0.13333333 0.08627451 0.0745098 ],  [0.13333333 0.08627451 0.0745098 ]],, [[0.17254902 0.11764706 0