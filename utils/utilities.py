import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.u_net_base import unet_base


def unsqueeze(input_numpy):
    return np.reshape(input_numpy, ((1,)+input_numpy.shape))

def crop_img(input_img_np, padding = 0):
    height, width, ch = input_img_np.shape
    cropped_img = input_img_np[height*2//3:, padding:width-padding, :]
    return cropped_img

def cc_map_postprocessing(input_img_np, original_size = (1080, 1920), padding = 0):
    original_height, original_width = original_size
    cropped_height_start_y = original_height*2//3
    recon_img = np.zeros(shape=(original_height, original_width, 1))
    recon_img[cropped_height_start_y:, padding:original_width-padding, :] = input_img_np
    return recon_img

def numpyIMG_resize(input_img_np, resize_shape = (240, 240)):
    height, width, ch = input_img_np.shape
    resize_img = tf.keras.utils.array_to_img(input_img_np).resize(resize_shape, Image.BICUBIC)
    # resize_img.show()
    return tf.keras.utils.img_to_array(resize_img)/255., height, width

def min_max_norm(input_numpy):
    normalized_np = 255*(input_numpy - np.min(input_numpy)) / (np.max(input_numpy) - np.min(input_numpy))
    return np.array(normalized_np, dtype=np.uint8)