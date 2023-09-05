import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.figure()
def get_input_data(file_path = "/Users/iptvpeullaespomgaebaltim/Documents/pythoncode/subscription_finder/datasets/ScaryMovie4.mp4"):
    vidcap = cv2.VideoCapture(file_path)
    vid_spf = 1/vidcap.get(cv2.CAP_PROP_FPS)
    frame_num = 0

    success, image = vidcap.read()
    time_stamp = 0
    while success and time_stamp<20:
        frame_num += 1
        success, image = vidcap.read()
        time_stamp = frame_num * vid_spf
        plt.imshow(image)
        plt.

        yield image
        # cropped_img = crop_img(image)
def crop_img(input_img_np, padding = 200):
    height, width, ch = input_img_np.shape
    cropped_img = input_img_np[height*2//3:, padding:width-padding, :]
    cropped_img = numpy_to_img(cropped_img)
    return cropped_img

def numpy_to_img(input_img_np):
    resize_img = tf.keras.utils.array_to_img(input_img_np).resize((240, 240), Image.BICUBIC)
    # resize_img.show()
    return tf.keras.utils.img_to_array(resize_img)

