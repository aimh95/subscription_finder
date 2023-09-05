import tensorflow as tf
import random
import glob
import string
import os
import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import tensorflow

crop_w = 240
crop_h = 240

def subscription_info_randgen(height, width):
    fonts_list = glob.glob("/Users/iptvpeullaespomgaebaltim/Documents/pythoncode/subscription_finder/utils/fonts/*.ttf")
    font_selection = random.choice(fonts_list)

    num_of_lines = random.randrange(1, 3)
    font_size = random.randrange(width//16, width//8)
    y_min = random.randrange(0, height-num_of_lines*font_size)
    # x_min = random.randrange(0, width-)
    subscription_sentences = []
    for line in range(num_of_lines):
        num_of_words = random.randrange(1, 8)
        len_of_word = [random.randrange(3, 8) for i in range(num_of_words)]

        sentence = ''
        letters = string.ascii_letters
        for word in range(num_of_words):
            # print(''.join(random.choice(letters) for i in range(len_of_word[word])), end= ' ')
            sentence += ''.join(random.choice(letters) for i in range(len_of_word[word]))
            sentence += ' '
        subscription_sentences.append(sentence)
    return subscription_sentences, y_min, num_of_lines, font_selection, font_size

def data_generator(input_img):
    height, width, ch = input_img.shape
    subscriptions, y_min, num_of_lines, font_selection, font_size = subscription_info_randgen(height, width)

    input_img_pil = tf.keras.utils.array_to_img(input_img)
    draw = ImageDraw.Draw(input_img_pil)
    font = ImageFont.truetype(font_selection, font_size)
    for i in range(len(subscriptions)):
        x_coord = (width - len(subscriptions[i]))//2
        draw.text((x_coord, y_min+i*font_size), subscriptions[i], fill='white', font=font)
    # input_img_pil.show()
    input_img_pil = subscription_crop(input_img_pil, y_min)

    return tf.keras.utils.img_to_array(input_img_pil), 1 if font_size>0 else 0

def subscription_crop(input_img, y_min):
    global crop_w, crop_h
    width, height = input_img.size
    move_coord_min = max(0, random.randrange(-height//8, 10))
    move_coord_max = crop_h + move_coord_min
    output_img = input_img.crop((0, max(y_min+move_coord_min, 0), width, min(y_min+move_coord_max, height)))
    output_img = output_img.resize((crop_w, crop_h), Image.BICUBIC)
    # output_img.show()
    return output_img

def custom_dataloader(data_path="/Users/iptvpeullaespomgaebaltim/Documents/pythoncode/subscription_finder/datasets/DIV2K_train_HR"):
    train_ds = tf.keras.utils.image_dataset_from_directory(data_path, labels=None, batch_size=1, image_size=(720, 1080), seed = 123)
    dataset_x, dataset_y = np.empty(shape=(len(train_ds), crop_h, crop_w, 3)), np.empty(shape=len(train_ds), dtype='int64')
    for i, x in enumerate(train_ds):
        subscription_flag = random.randrange(0, 2)
        if subscription_flag:
            dataset_x[i], dataset_y[i] = data_generator(x[0])
        else:
            dataset_x[i], dataset_y[i] = subscription_crop(tf.keras.utils.array_to_img(x[0]), random.randrange(0, 600)), 0
    return (dataset_x, dataset_y)

custom_dataloader()
