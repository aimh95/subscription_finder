import numpy as np
import glob
import random
import string
import tensorflow as tf
from PIL import ImageFont, ImageDraw
import matplotlib.pyplot as plt
from utils.utilities import random_crop
import cv2
import utils.utilities as util

def character_randgen(height, width):
    fonts_list = glob.glob("/Users/pythoncodes/subtitle_finder/utils/fonts/*.ttf")

    num_of_char = random.randrange(0, 8)


    # char_list, y_min, x_min, y_max, x_max = [], [], [], [], []
    # font_selection, font_size = [], []

    char_list = []

    for char in range(num_of_char):
        char_info = dict()
        char_info["character"] = random.choice(string.ascii_letters)

        char_info["font_selection"] = random.choice(fonts_list)
        char_info["font_size"] = random.randrange(width // 16, width // 4)
        font = ImageFont.truetype(char_info["font_selection"], char_info["font_size"])
        char_width, char_height = font.getsize(char_info["character"])

        char_info["fill_color"] = (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))
        char_info["stroke_fill"] = (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))
        char_info["stroke_width"] = random.randrange(0, 5)
        char_info["y_min"] = random.randrange(0, height-char_height)
        char_info["x_min"] = random.randrange(0, width-char_width)

        char_info["y_max"] = char_info["y_min"] + char_height
        char_info["x_max"] = char_info["x_min"] + char_width
        char_list.append(char_info)

    return char_list


def sentence_randgen(height, width):
    fonts_list = glob.glob("/Users/pythoncodes/subtitle_finder/utils/fonts/*.ttf")

    num_of_char = random.randrange(0, 16)

    char_info = dict()
    sentence = ""
    for i in range(num_of_char):
        sentence+=random.choice(string.ascii_letters)
    char_info["sentence"] = sentence
    char_info["font_selection"] = random.choice(fonts_list)
    char_info["font_size"] = random.randrange(width // 16, width // 4)

    char_info["fill_color"] = (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))
    char_info["stroke_fill"] = (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))
    char_info["stroke_width"] = random.randrange(0, 2)
    char_info["y_min"] = random.randrange(0, height-char_info["font_size"])
    char_info["x_min"] = random.randrange(0, width//16)

    return char_info


def sentence_gaussian_mapping(input_numpy, sentence):
    height, width, ch = input_numpy.shape
    font = ImageFont.truetype(sentence["font_selection"], sentence["font_size"])
    x_coord, y_coord = sentence["x_min"], sentence["y_min"]
    gaussian_map = np.zeros((height, width))

    for char in sentence["sentence"]:
        char_width, char_height = font.getsize(char)
        x_cent = x_coord + char_width//2
        y_cent = y_coord + char_height//2



        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        # y = x[:, np.newaxis]x
        # y = x[:,np.arange(0, y_max, 1, float)]

        y_sigma = char_height
        x_sigma = char_width
        x_coord = x_coord + char_width
        # y_coord = y_coord + char_height

        gaussian_map += np.exp(-4 * np.log(2) * (((x - x_cent)/x_sigma)**2  + ((y - y_cent)/ y_sigma)**2))

    return gaussian_map

def gaussian_distribution(size, fwhm, center=None):
    # g(x,y) = (1/2pi*sig^2)exp(-(x^2+y^2)/2sig^2)
    #tilting = [cos -sin sin cos]
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def character_gaussian_mapping(input_numpy, char):
    height, width, ch = input_numpy.shape
    x_cent = (char["x_max"] + char["x_min"]) // 2
    y_cent = (char["y_max"] + char["y_min"]) // 2

    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    # y = x[:, np.newaxis]x
    # y = x[:,np.arange(0, y_max, 1, float)]

    sigma = char["y_max"] - char["y_min"]

    gaussian_distribution = np.exp(-4 * np.log(2) * ((x - x_cent) ** 2 + (y - y_cent) ** 2) / sigma ** 2)


    return gaussian_distribution


def dataset_sentence_generator(input_img):
    # plt.figure()

    height, width, ch = input_img.shape
    sentence_info = sentence_randgen(height, width)
    input_img_pil = tf.keras.utils.array_to_img(input_img)
    gaussian_map = np.zeros((height, width))
    draw = ImageDraw.Draw(input_img_pil)
    font = ImageFont.truetype(sentence_info["font_selection"], sentence_info["font_size"])
    # draw.text((sentence_info["x_min"], sentence_info["y_min"]), sentence_info["sentence"], fill=sentence_info["fill_color"], font=font,stroke_width=sentence_info["stroke_width"], stroke_fill=sentence_info["stroke_fill"])
    draw.text((sentence_info["x_min"], sentence_info["y_min"]), sentence_info["sentence"], fill=sentence_info["fill_color"], font=font,
              stroke_width=sentence_info["stroke_width"], stroke_fill=sentence_info["stroke_fill"])
    # input_img_pil.show()
    gaussian_map = sentence_gaussian_mapping(tf.keras.utils.img_to_array(input_img_pil), sentence_info)
    # gaussian_map = np.clip(gaussian_map, 0, 1)
    # plt.figure()
    # plt.imshow(gaussian_map)

    # return tf.keras.utils.img_to_array(input_img_pil)/255., subtitle_mapping
    # rotate_angle = random.randrange(0, 360)
    # cX, cY = random.randrange(0, width), random.randrange(0, height)
    # output_img = util.rotate_img(tf.keras.utils.img_to_array(input_img_pil)/255.,  width//2, height//2, rotate_angle)
    # plt.imshow(output_img)
    # gaussian_map = util.rotate_img(gaussian_map, width//2, height//2, rotate_angle)
    # plt.imshow(gaussian_map)

    return tf.keras.utils.img_to_array(input_img_pil)/255., gaussian_map


def dataset_generator(input_img):
    # plt.figure()

    height, width, ch = input_img.shape
    char_list = character_randgen(height, width)
    input_img_pil = tf.keras.utils.array_to_img(input_img)
    gaussian_map = np.zeros((height, width))
    for char in char_list:
        draw = ImageDraw.Draw(input_img_pil)
        font = ImageFont.truetype(char["font_selection"], char["font_size"])
        draw.text((char["x_min"], char["y_min"]), char["character"], fill=char["fill_color"], font=font, stroke_width=char["stroke_width"], stroke_fill=char["stroke_fill"])
        gaussian_map += character_gaussian_mapping(tf.keras.utils.img_to_array(input_img_pil), char)
    gaussian_map = np.clip(gaussian_map, 0, 1)
    # plt.imshow(gaussian_map)
    # input_img_pil.show()
    # pass


    # return tf.keras.utils.img_to_array(input_img_pil)/255., subtitle_mapping
    return tf.keras.utils.img_to_array(input_img_pil)/255., gaussian_map


def custom_dataloader(data_path="/Users/pythoncodes/subtitle_finder/datasets/train_dataset/indoor",batch_size = 32, image_size = (480, 720), crop_size = 240):
    train_ds = tf.keras.utils.image_dataset_from_directory(data_path, labels=None, batch_size=1, image_size=image_size)
    n_batch = len(train_ds)//batch_size

    train_x = np.empty(shape=(len(train_ds), crop_size, crop_size, 3))
    train_y = np.empty(shape=(len(train_ds), crop_size, crop_size, 1))

    # plt.figure()
    for i, img in enumerate(train_ds):
        img = random_crop(img[0], crop_size)
        train_x[i], train_y[i, :, :, 0] = dataset_sentence_generator(img)
        plt.imshow(train_x[i])
        plt.imshow(train_y[i])


    return train_x, train_y

custom_dataloader()