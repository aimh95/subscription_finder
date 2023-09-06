import keras.utils
import tensorflow as tf
import random
import glob
import string
import os
import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import tensorflow
import matplotlib.pyplot as plt

def subscription_info_randgen(height, width):
    fonts_list = glob.glob("/Users/iptvpeullaespomgaebaltim/Documents/pythoncode/subscription_finder/utils/movie_fonts/*.ttf")
    font_selection = random.choice(fonts_list)

    num_of_lines = random.randrange(1, 2)
    font_size = random.randrange(width//32, width//8)
    y_min = random.randrange(0, height-num_of_lines*font_size)
    subscription_sentences = []
    max_string_width = 0

    max_len = 0
    for line in range(num_of_lines):
        num_of_words = random.randrange(1, 8)
        len_of_word = [random.randrange(3, 8) for i in range(num_of_words)]

        sentence = ''
        letters = string.ascii_letters
        for word in range(num_of_words):
            # print(''.join(random.choice(letters) for i in range(len_of_word[word])), end= ' ')
            sentence += ''.join(random.choice(letters) for i in range(len_of_word[word]))
            sentence += ' '
        font = ImageFont.truetype(font_selection, font_size)
        sentence_width, sentence_height = font.getsize(sentence)
        max_len = max(max_len, sentence_width)
        subscription_sentences.append(sentence)
    x_min = random.randrange(0, width//2)
    x_max = int(min(x_min + max_len, width))
    y_max = min(y_min + num_of_lines * sentence_height, height)

    return subscription_sentences, x_min, y_min, x_max, y_max, font_selection, font_size

def ground_truth_mapping(input_img, x_min, y_min, x_max, y_max):
    output_img = np.zeros(shape=input_img.shape)
    output_img[y_min:y_max, x_min:x_max, :] = 1
    # keras.utils.array_to_img(input_img).show()
    # keras.utils.array_to_img(input_img*output_img).show()
    # keras.utils.array_to_img(input_img[y_min:y_max, x_min:x_max, :]).show()
    return output_img

def dataset_generator(input_img):
    height, width, ch = input_img[0].shape
    subscriptions, x_min, y_min, x_max, y_max, font_selection, font_size = subscription_info_randgen(height, width)

    input_img_pil = tf.keras.utils.array_to_img(input_img[0])
    draw = ImageDraw.Draw(input_img_pil)
    font = ImageFont.truetype(font_selection, font_size)
    for i in range(len(subscriptions)):
        draw.text((x_min, y_min+i*font_size), subscriptions[i], fill='white', font=font)
    subscription_mapping = ground_truth_mapping(keras.utils.img_to_array(input_img_pil), x_min, y_min, x_max, y_max)
    # input_img_pil.show()

    return tf.keras.utils.img_to_array(input_img_pil)/255., subscription_mapping

def crop_img(input_img, x_min, y_min, x_max, y_max):
    input_img = tf.keras.utils.array_to_img(input_img)
    output_img = input_img.crop((x_min, y_min, x_max, y_max))
    return tf.keras.utils.img_to_array(output_img)

def img_to_grid(input_img, n_sep=16):
    # plt.figure()
    height, width, _ = input_img.shape

    crop_height = height//n_sep
    crop_width = width//n_sep

    output_grid = np.empty(shape=(n_sep**2, crop_height, crop_width, 3))
    # plt.figure()
    # plt.imshow(input_img)
    for i in range(n_sep):
        for j in range(n_sep):
            output_grid[i*n_sep+j] = input_img[i*crop_height:(i+1)*crop_height, j*crop_width:(j+1)*crop_width, :]   #crop_img(input_img, i*crop_height, j*crop_width, (i+1)*crop_height, (j+1)*crop_width)
            # plt.imshow(output_grid[i])
    # fig = plt.figure()
    # rows = 5
    # cols = 5
    # for i in range(8192):
    #     ax = fig.add_subplot(rows, cols, i + 1)
    #     ax.imshow(output_grid[i])
    #     pass
    return output_grid

def grid_to_img(input_grid, n_sep=16):
    plt.figure()
    n_grid, crop_height, crop_width, _ = input_grid.shape
    output_img = np.empty(shape=(1, crop_height*n_sep, crop_width*n_sep, 3))
    for i in range(n_sep):
        for j in range(n_sep):
            output_img[i*crop_height:(i+1)*crop_height, j*crop_width:(j+1)*crop_width, :] = input_grid[i]
    return output_img

def custom_dataloader(data_path="/Users/iptvpeullaespomgaebaltim/Documents/pythoncode/subscription_finder/datasets/DIV2K_train_HR", n_sep = 5, batch_size = 32):
    train_ds = tf.keras.utils.image_dataset_from_directory(data_path, labels=None, batch_size=1, image_size=(720, 1080), seed = 123)

    n_grid = n_sep**2
    n_grids = n_grid*len(train_ds)
    crop_height = 720//n_sep
    crop_width = 1080//n_sep
    n_batch = len(train_ds)//batch_size

    # train_x = np.empty(shape=(n_batch, n_grid*batch_size, crop_height, crop_width, 3))
    # train_y = np.empty(shape=(n_batch, n_grid*batch_size, crop_height, crop_width, 3))
    train_x = np.empty(shape=(n_grids, crop_height, crop_width, 3))
    train_y = np.empty(shape=(n_grids, crop_height, crop_width, 3))
    for i, image in enumerate(train_ds):
        dataset_x, dataset_y = dataset_generator(image)
        # train_x[i // batch_size, i % batch_size * n_grid:(i % batch_size + 1) * n_grid] = img_to_grid(dataset_x, n_sep)
        # train_y[i // batch_size, i % batch_size * n_grid:(i % batch_size + 1) * n_grid] = img_to_grid(dataset_y, n_sep)
        train_x[i * n_grid:(i + 1) * n_grid] = img_to_grid(dataset_x, n_sep)
        train_y[i * n_grid:(i + 1) * n_grid] = img_to_grid(dataset_y, n_sep)
    # fig = plt.figure()
    # rows = 5
    # cols = 5
    # for i in range(25):
    #     ax = fig.add_subplot(rows, cols, i + 1)
    #     ax.axis("off")
    #     ax.imshow(train_x[i+25])
    #     pass
    return train_x, train_y


# custom_dataloader()
