import random

import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.u_net_base import unet_base


def unsqueeze(input_numpy):
    return np.reshape(input_numpy, ((1,) + input_numpy.shape))


def crop_img(input_img_np, padding=0):
    height, width, ch = input_img_np.shape
    cropped_img = input_img_np[height * 2 // 3:, padding:width - padding, :]
    return cropped_img


def random_crop(input_img_np, crop_size):
    height, width, ch = input_img_np.shape
    x = random.randrange(0, width - crop_size)
    y = random.randrange(0, height - crop_size)
    cropped_img = input_img_np[y:y + crop_size, x:x + crop_size]
    return cropped_img


def cc_map_postprocessing(input_img_np, original_size=(1080, 1920), padding=0):
    original_height, original_width = original_size
    cropped_height_start_y = original_height * 2 // 3
    recon_img = np.zeros(shape=(original_height, original_width, 1))
    recon_img[cropped_height_start_y:, padding:original_width - padding, :] = input_img_np
    return recon_img


def numpyIMG_resize(input_img_np, resize_shape=(240, 240)):
    height, width, ch = input_img_np.shape
    resize_img = tf.keras.utils.array_to_img(input_img_np).resize(resize_shape, Image.BICUBIC)
    # resize_img.show()
    return tf.keras.utils.img_to_array(resize_img) / 255., height, width


def min_max_norm(input_numpy):
    normalized_np = 255 * (input_numpy - np.min(input_numpy)) / (np.max(input_numpy) - np.min(input_numpy))
    return np.array(normalized_np, dtype=np.uint8)


# sobel operation -> edge to black
def sobel_operation(input_numpy):
    dx = cv2.Sobel(input_numpy, -1, 1, 0)
    dy = cv2.Sobel(input_numpy, -1, 0, 1)
    mag = cv2.magnitude(dx, dy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)

    cv2.imshow("original", input_numpy)
    # cv2.imshow("dx", np.abs(dx))
    # cv2.imshow("dy", np.abs(dy))
    # cv2.imshow("dxdy", np.abs(dx)+np.abs(dy))
    cv2.waitKey(1)
    return


def step_decay(epoch):
    start = 0.1
    drop = 0.5
    epochs_drop = 5.0
    lr = start * (drop ** np.floor((epoch) / epochs_drop))
    return lr


def rotate_img(input_numpy, cX, cY, rotate_angle):
    height, width = input_numpy.shape[:2]

    M = cv2.getRotationMatrix2D((cX, cY), rotate_angle, 1.0)
    rotated_img = cv2.warpAffine(input_numpy, M, (width, height))
    return rotated_img


def grad(model, inputs, targets, loss):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def get_cnt_from_character_prob(prob_heatmap):
    ret, binary_gaussian_map = cv2.threshold(prob_heatmap, 0.5, 1, cv2.THRESH_BINARY)
    plt.figure("binary_gaussian_map")
    plt.imshow(binary_gaussian_map)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.morphologyEx(binary_gaussian_map, cv2.MORPH_OPEN, kernel, iterations=0)
    distance_transform = cv2.distanceTransform((binary_gaussian_map * 255.).astype('uint8'), cv2.DIST_L2, 3)

    ret, sure_fg = cv2.threshold(distance_transform, 0.5 * distance_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = np.logical_xor(sure_fg, sure_bg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    return max(markers) - 1, markers
