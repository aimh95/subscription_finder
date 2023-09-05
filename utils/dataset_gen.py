import tensorflow as tf
import random
import glob
import string
import os
import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np

def subscription_gen(height, width):
    fonts_list = glob.glob("./fonts/*.ttf")
    font_selection = random.choice(fonts_list)
    font_size = random.randrange(width//20, width//10)
    num_of_words = random.randrange(1, width//(font_size*5))
    len_of_word = [random.randrange(3, 5) for i in range(num_of_words)]
    cc_length = font_size*sum(len_of_word)//num_of_words + font_size*(num_of_words-1)//2
    x_min, y_min = random.randrange(0, width-cc_length), random.randrange(0, height-font_size)
    x_max, y_max = x_min + cc_length, y_min + font_size
    output_string = ''
    letters = string.ascii_letters
    for word in range(num_of_words):
        # print(''.join(random.choice(letters) for i in range(len_of_word[word])), end= ' ')
        output_string += ''.join(random.choice(letters) for i in range(len_of_word[word]))
        output_string += ' '
    print(output_string)
    return output_string, x_min, y_min, x_max, y_max, font_selection, font_size

def load_image(image_dir = "../datasets/single_image/109666.jpeg"):
    input_img = cv2.imread(image_dir)
    input_img = cv2.resize()
    input_img_norm = input_img/255.
    height, width, ch = input_img.shape
    for i in range(100):
        subscript, x_min, y_min, x_max, y_max, font_selection, font_size = subscription_gen(height, width)
        custom_dataloader(input_img, subscript, x_min, y_min, x_max, y_max, font_selection, font_size)

def data_generator(input_img):
    height, width, ch = input_img.shape
    subscript, x_min, y_min, x_max, y_max, font_file, font_size = subscription_gen(height, width)

    input_img_pil = tf.keras.utils.array_to_img(input_img)
    draw = ImageDraw.Draw(input_img_pil)
    font = ImageFont.truetype(font_file, font_size)
    draw.text((x_min, y_min), subscript, fill='white', font=font)
    input_img_pil.show()
    subscription_crop(input_img_pil, x_min, y_min, x_max, y_max)

    return tf.keras.utils.img_to_array(input_img_pil), 1 if font_size>0 else 0

def subscription_crop(input_img, x_min, y_min, x_max, y_max):
    # height, width, _ = input_img.size
    output_img = input_img.crop((max(x_min+random.randrange(-20, 1), 0), max(y_min+random.randrange(-20, 1), 0), min(x_max+random.randrange(-1, 20), 255), min(y_max+random.randrange(-1, 20), 255)))
    output_img.show()
    pass

def custom_dataloader(data_path = "../datasets/val2017", batch_size = 32):
    train_ds = tf.keras.utils.image_dataset_from_directory(data_path, labels=None, batch_size=batch_size, seed = 123)
    output_x, output_y = False, []
    for i, x in enumerate(train_ds):
        for image in range(batch_size):
            height, width, ch = x[image].shape
            temp_x, temp_y = data_generator(x[image])
            tf.keras.utils.array_to_img(x[image]).show()
            if type(output_x) == bool:
                output_x, output_y = temp_x / 255., temp_y
            else:
                output_x = np.append(output_x, temp_x / 255., axis=0)
                output_y.append(temp_y)
    return output_x, output_y
custom_dataloader()
