import numpy as np
import tensorflow as tf
import math
import os
import cv2
import glob

class CustomDataLoader(tf.keras.utils.Sequence):
    def __init__(self, is_train=1, batch_size=32, data_dir="/Users/iptvpeullaespomgaebaltim/Documents/pythoncode/subscription_finder/datasets/DIV2K_train_HR"):
        self.batch_size = batch_size
        self.is_train = is_train
        self.image_list = glob.glob(data_dir)
        self.dataset = []

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.dataset))
        for _, img_name in enumerate(self.image_list):
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.dataset.append(img)
        batch_x = self.dataset[low:high]
        return 1

    def make_grid(self, input_batch, ):
        pass

train_gen = CustomDataLoader()
train_gen.__len__()
pass