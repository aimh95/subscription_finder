import tensorflow as tf
import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from keras.preprocessing.image import img_to_array, array_to_img
from utils.custom_utils import get_cnt_from_character_prob


dataset_path = "../datasets/train_dataset/totaltext/train_pixel"
files = os.listdir(dataset_path)

for file in files:
    text_pixel_groundtruth = cv2.imread(os.path.join(dataset_path, file))[:, :, 0]
    cv2.imshow("", text_pixel_groundtruth)
    ret, binary_gaussian_map = cv2.threshold(text_pixel_groundtruth, 125, 255, cv2.THRESH_BINARY)
    cv2.imshow("", binary_gaussian_map)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary_gaussian_map, kernel, iterations=3)
    sure_bg = cv2.morphologyEx(sure_bg, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow("", sure_bg)
    distance_transform = cv2.distanceTransform(np.uint8(binary_gaussian_map), cv2.DIST_L2, 1)
    cv2.imshow("", distance_transform)
    ret, sure_fg = cv2.threshold(distance_transform, 0.1 * distance_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    cv2.imshow("", sure_fg)

    unknown = np.logical_xor(sure_fg, sure_bg)
    plt.figure()
    plt.imshow(unknown)
    cv2.imshow("", unknown)
    ret, markers = cv2.connectedComponents(sure_fg)
    plt.imshow(markers)

