import random
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import cv2
import os
import glob
from models.real_u_net import U_Net
import string
from tensorflow.python.data.experimental import AUTOTUNE
import numpy as np
import math
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import matplotlib.pyplot as plt
from utils.custom_utils import unsqueeze

class CustomDataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size = 32, target_resolution = (240, 240), shuffle=False, heatmap_flag="gaussian"):
        super(CustomDataLoader, self).__init__()
        # self.dataset_path = dataset_path
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.resolution = target_resolution
        self.data_dir = dataset_path

        self.image_numpy_list = self._image_dataset(self._image_file())
        self.heatmap_flag= heatmap_flag

        self.batch_num_of_char = []

        # self.dataset = self._image_dataset_gen()

        # self.ds = self.dataset()

    def __len__(self):
        return math.ceil(len(self.image_numpy_list) / self.batch_size)

    def __getitem__(self, idx):
        # print("__getitem__ 접근", idx)
        batch_x, batch_y = self._image_dataset_gen(self.image_numpy_list[idx*self.batch_size:(idx+1)*self.batch_size])
        return batch_x, batch_y

    def _dataset(self):
        ds = self._image_dataset(self.image_loaded, ch=3)
        return ds

    def _subtitle_info_randgen(self):
        fonts_list = glob.glob("../utils/movie_fonts/*.ttf")

        subtitle_sentences = dict()

        max_len = 0
        num_of_words = random.randrange(1, 8)
        len_of_word = [random.randrange(3, 8) for i in range(num_of_words)]
        # self.batch_num_of_char.append(np.sum(len_of_word))
        sentence = ''
        letters = string.ascii_letters
        for word in range(num_of_words):
            # print(''.join(random.choice(letters) for i in range(len_of_word[word])), end= ' ')
            sentence += ''.join(random.choice(letters) for i in range(len_of_word[word]))
            sentence += ' '

        font_selection = random.choice(fonts_list)
        font_size = random.randrange(self.resolution[1] // 32, self.resolution[1] // 8)

        self.font = ImageFont.truetype(font_selection, font_size)
        # subtitle_sentences["font"]= ImageFont.truetype(font_selection, font_size)
        sentence_width, sentence_height = self.font.getsize(sentence)
        x_min = random.randrange(0, max(1, self.resolution[1] - sentence_width))
        y_min = random.randrange(0, max(1, self.resolution[0] - sentence_height))
        x_max = int(min(x_min + sentence_width, self.resolution[1]))
        y_max = int(min(y_min + sentence_height, self.resolution[0]))

        fill_color = (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))
        stroke_color = (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))
        stroke_width = random.randrange(0, 2)

        subtitle_sentences["sentence"] = sentence
        subtitle_sentences["x_min"] = x_min
        subtitle_sentences["y_min"] = y_min
        subtitle_sentences["x_max"] = x_max
        subtitle_sentences["y_max"] = y_max
        subtitle_sentences["font_selection"] = font_selection
        subtitle_sentences["font_size"] = font_size
        subtitle_sentences["fill_color"] = fill_color
        subtitle_sentences["stroke_color"] = stroke_color
        subtitle_sentences["stroke_width"] = stroke_width


        return subtitle_sentences

    def _image_name_list(self):
        image_dir = self.data_dir
        image_list = [file_name.split(".")[-2] for file_name in os.listdir(image_dir)]
        return image_list

    def _get_extension(self, file_dir):
        return os.path.splitext(os.listdir(file_dir)[0])[-1]

    def _image_file(self):
        image_dir = self.data_dir
        file_extension = self._get_extension(image_dir)
        image_list = [os.path.join(image_dir, file+file_extension) for file in self._image_name_list()]
        return image_list

    # @staticmethod
    def _image_dataset(self, image_files):
        # plt.figure()
        image_list = []
        for i, file in enumerate(image_files):
            self.random_sentence = self._subtitle_info_randgen()
            image_list.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))
            # plt.imshow(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))
        return image_list
    # plt.figure()
    def _image_dataset_gen(self, image_batch):
        x_dataset = np.empty(shape=(len(image_batch), self.resolution[0], self.resolution[1], 3))
        y_dataset = np.empty(shape=(len(image_batch), self.resolution[0], self.resolution[1], 1))
        for i, image_numpy in enumerate(image_batch):
            self.random_sentence = self._subtitle_info_randgen()
            image = self._random_crop(image_numpy)
            # plt.imshow(image)
            image = self._sentence_burn_in(image)
            # plt.imshow(image)
            if self.heatmap_flag == "binary":
                mapping = self._sentence_map()
            elif self.heatmap_flag == "gaussian":
                mapping = self._sentence_gaussian_mapping(image, self.random_sentence)[..., tf.newaxis]
            # plt.imshow(mapping)
            x_dataset[i], y_dataset[i] = image, mapping

        return x_dataset, y_dataset

    def _unsqueeze(self, image):
        height, width = image.shape
        return image.reshape((height, width, 1))

    def _random_crop(self, image):
        height, width, ch = image.shape
        dy, dx = self.resolution

        y = random.randrange(0, max(1, height-dy))
        x = random.randrange(0, max(1, width-dx))

        cropped_img = self._pedded_img(image[y:min(height, y+dy), x:min(width, x+dx), :])
        return cropped_img

    def _pedded_img(self, image):
        y, x, ch = image.shape
        padded_img = np.zeros((self.resolution[0], self.resolution[1], 3))
        padded_img[:y, :x, :] = image
        return padded_img

    def _sentence_burn_in(self, image):
        image_pil = array_to_img(image)
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype(self.random_sentence["font_selection"], self.random_sentence["font_size"])
        draw.text((self.random_sentence["x_min"], self.random_sentence["y_min"]), self.random_sentence["sentence"],
                  fill="white", font=font, stroke_width=self.random_sentence["stroke_width"], stroke_fill="black")
        # image_pil.show()
        return img_to_array(image_pil)/255.

    def _sentence_map(self):
        sentence_map = np.zeros(shape=(self.resolution[0], self.resolution[1], 1))
        sentence_map[self.random_sentence["y_min"]:self.random_sentence["y_max"],
                     self.random_sentence["x_min"]:self.random_sentence["x_max"], :] = 1
        return sentence_map

    # def _sentence_gaussian_distribution_map(self):
    #     x_start, y_start = self.random_sentence["x_min"], self.random_sentence["y_min"]
    #     sentence_map = np.zeros(shape=(self.resolution[0], self.resolution[1], 1))
    #     for char in self.random_sentence["sentence"]:
    #         sentence_width, sentence_height = self.font.getsize(char)
    #         char_x_center, char_y_center = x_start + (sentence_width)//2, y_start + (sentence_height)//2
    #         x_start += sentence_width
    #         max_ = max(sentence_width, sentence_height)
    #         sentence_map[x_start:x_start+max_, y_start:y_start+max_, 0] += self.sentence_gaussian_mapping()
    #
    #         plt.figure()
    #         plt.imshow(sentence_map, cmap='summer')
    #         plt.colorbar()
    #         plt.show()
    #         pass

    def _sentence_gaussian_mapping(self, input_numpy, sentence):
        height, width, ch = input_numpy.shape
        font = ImageFont.truetype(sentence["font_selection"], sentence["font_size"])
        x_coord, y_coord = sentence["x_min"], sentence["y_min"]
        gaussian_map = np.zeros((height, width))

        for char in sentence["sentence"]:
            char_width, char_height = font.getsize(char)
            if char == ' ':
                x_coord = x_coord + char_width
            else:

                x_cent = x_coord + char_width // 2
                y_cent = y_coord + char_height // 2

                x = np.arange(0, width, 1, float)
                y = np.arange(0, height, 1, float)[:, np.newaxis]

                y_sigma = char_height
                x_sigma = char_width
                x_coord = x_coord + char_width

                gaussian_map += np.exp(-8 * np.log(2) * (((x - x_cent) / x_sigma) ** 2 + ((y - y_cent) / y_sigma) ** 2))
        return gaussian_map


# dataloader = CustomDataLoader("../datasets/train_dataset/indoor")
# x, y = dataloader.data_dir()