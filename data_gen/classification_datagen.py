import tensorflow as tf
import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import math

class TotalTextClassificationDataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size=32, crop_size = (240, 240), shuffle=False):
        super(TotalTextClassificationDataLoader, self).__init__()
        # self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.crop_size = crop_size
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
        x_dataset = np.empty(shape=(self.batch_size, self.crop_size[0], self.crop_size[1], 3))
        y_dataset = np.empty(shape=(self.batch_size))

        for i in range(len(x_input_list)):
            random_coord = self._get_random_coord(x_input_list[i])

            x_data = self._image_crop(x_input_list[i], coord=random_coord)
            y_data = self._image_crop(y_input_list[i], coord=random_coord)
            # plt.figure()
            # plt.imshow(x_data/255)
            # plt.imshow(y_data)

            x_dataset[i] = x_data
            y_dataset[i] = 1 if np.sum(y_data)>0 else 0
        return x_dataset, y_dataset

    def _get_random_coord(self, image):
        height, width, ch = image.shape
        y_coord = random.randint(0, max(1,height-self.crop_size[0]))
        x_coord = random.randint(0, max(1, width-self.crop_size[1]))
        return (y_coord, x_coord)

    def _image_crop(self, image, coord):
        y, x = coord
        dy, dx = self.crop_size
        cropped_img = image[y:min(image.shape[0], y+dy), x:min(image.shape[1], x+dx), :]
        cropped_img = self._img_padding(cropped_img)
        return cropped_img

    def _img_padding(self, image):
        y, x, ch = image.shape
        padded_img = np.zeros((self.crop_size[0], self.crop_size[1], 3))
        padded_img[:y, :x, :] = image
        return padded_img

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



