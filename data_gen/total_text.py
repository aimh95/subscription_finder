import tensorflow as tf
import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import math

class TotalTextDataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size=32, target_resolution = (240, 240), shuffle=False):
        super(TotalTextDataLoader, self).__init__()
        # self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.resolution = target_resolution
        self.x_dir = os.path.join(dataset_path, "train_x")
        self.y_dir = os.path.join(dataset_path, "train_y")

        self.x = self.x_dataset()
        self.y = self.y_dataset()
        # self.ds = self.dataset()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x, batch_y = self._image_dataset_gen(batch_x, batch_y)
        return batch_x, batch_y

    def _image_dataset_gen(self, x_input_list, y_input_list):
        # print("__getitem__ 접근", idx)
        x_dataset = np.empty(shape=(self.batch_size, self.resolution[0], self.resolution[1], 3))
        y_dataset = np.empty(shape=(self.batch_size, self.resolution[0], self.resolution[1], 1))

        for i in range(len(x_input_list)):
            x_data = self._resize_img(x_input_list[i])
            y_data = self._resize_img(y_input_list[i])
            # plt.figure()
            # plt.imshow(x_data)
            # plt.imshow(y_data)
            # could not broadcast input array from shape (240,240) into shape (240,240,1)
            x_dataset[i] = x_data
            y_dataset[i] = y_data

pass
        return x_dataset, y_dataset

    # def _image_dataset_gen(self, x_input_list, y_input_list):
    #     # print("__getitem__ 접근", idx)
    #     try:
    #         x_dataset = np.empty(shape=(self.batch_size, self.resolution[0], self.resolution[1], 3))
    #         y_dataset = np.empty(shape=(self.batch_size, self.resolution[0], self.resolution[1], 1))
    #
    #         for i in range(self.batch_size):
    #             height, width = x_input_list[i].shape[0], x_input_list[i].shape[1]
    #
    #             dy, dx = self.resolution
    #
    #             y = random.randrange(0, max(1, height - dy))
    #             x = random.randrange(0, max(1, width - dx))
    #             self.random_coord = (y, x)
    #             x_data = self._image_crop(x_input_list[i])
    #             y_data = self._image_crop(y_input_list[i])
    #             data_width, data_height, _ = x_data.shape
    #             if data_width < dx or data_height < dy:
    #                 x_data = self._padding_img(x_data)
    #                 y_data = self._padding_img(y_data)
    #
    #             #could not broadcast input array from shape (240,240) into shape (240,240,1)
    #             x_dataset[i] = x_data/255.
    #             y_dataset[i] = y_data/255.
    #     except:
    #         pass
    #         # plt.figure()
    #         # plt.imshow(x_dataset[i])
    #         # plt.imshow(y_data)
    #     return x_dataset, y_dataset

    def _image_crop(self, image):
        y, x = self.random_coord
        dy, dx = self.resolution
        cropped_img = image[y:y+dy, x:x+dx, :]
        return cropped_img

    def data_gen(self):
        return self.x, self.y

    def x_dataset(self):
        x_data_list = self._image_dataset(self._x_image_file(), ch=3)
        return x_data_list

    def y_dataset(self):
        y_data_list = self._image_dataset(self._y_image_file(), ch=1)
        return y_data_list

    def _image_name_list(self):
        image_dir = self.x_dir
        image_list = [file_name.split(".")[-2] for file_name in os.listdir(image_dir)]
        return image_list

    def _get_extension(self, file_dir):
        return os.path.splitext(os.listdir(file_dir)[0])[-1]

    def _x_image_file(self):
        image_dir = self.x_dir
        file_extension = self._get_extension(image_dir)
        image_list = [os.path.join(image_dir, file+file_extension) for file in self._image_name_list()]
        return image_list

    def _y_image_file(self):
        image_dir = self.y_dir
        file_extension = self._get_extension(image_dir)
        image_list = [os.path.join(image_dir, file+file_extension) for file in self._image_name_list()]
        return image_list

    # @staticmethod
    def _image_dataset(self, image_files, ch):

        # ds = tf.data.Dataset.from_tensor_slices(image_files)
        # ds = ds.map(tf.io.read_file)
        # ds = ds.map(lambda x: tf.image.decode_image(x), num_parallel_calls=AUTOTUNE)
        image_len = len(image_files)
        image = []
        for i, file in enumerate(image_files):
            if ch == 1:
                image.append(self._unsqueeze(cv2.imread(file, cv2.IMREAD_GRAYSCALE)))
            else:
                image.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))
        return image

    def _unsqueeze(self, image):
        height, width = image.shape
        return image.reshape((height, width, 1))

    def _padding_img(self, image):
        height, width, ch = image.shape
        if height>=self.resolution[0] or width>=self.resolution[1]:
            image = self._resize_img(image)
            height, width, ch = image.shape
        padded_img = np.zeros(shape=(self.resolution[0], self.resolution[1], ch))
        padded_img[:height, :width, :] = image / np.max(image)
        return padded_img

    def _resize_img(self, image):
        output_image = cv2.resize(image, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_LANCZOS4)
        if len(output_image.shape) == 2:
            output_image = self._unsqueeze(output_image)
        return output_image / 255
