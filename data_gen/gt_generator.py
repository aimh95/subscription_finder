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
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math
from utils.custom_utils import unsqueeze
import csv
import pandas as pd

class ReadVideoAsData(tf.keras.utils.Sequence):
    def __init__(self, dataset_dir, output_dir, padding = 140, frame_size = (480, 720)):
        super(ReadVideoAsData, self).__init__()
        # self.dataset_path = dataset_path
        self.data_dir = dataset_dir
        self.output_dir = output_dir

        self.padding = padding

        self.frame_size = frame_size

        self.test_data_list = self._get_test_data_list()

    def _get_test_data_list(self):
        data_list = os.listdir(self.data_dir)
        return data_list

    def csv_write(self, arr, file_name):
        df = pd.DataFrame(arr, columns=["subtitle"])
        df.to_csv(os.path.join(self.output_dir, file_name.split(".")[0]+".csv"))

    def _save_frame_img(self, image, test_file, frame_number):
        image_pil = tf.keras.utils.array_to_img(image)
        file_dir = os.path.join(self.output_dir, test_file.split(".")[0])
        if os.path.isdir(file_dir):
            image_pil.save(os.path.join(self.output_dir, test_file.split(".")[0], "frame_num_"+str(frame_number)+".png"))
        else:
            os.mkdir(file_dir)
            image_pil.save(
                os.path.join(self.output_dir, test_file.split(".")[0], "frame_num_" + str(frame_number) + ".png"))
        # cv2.imwrite(os.path.join(self.output_dir, test_file.split(".")[0], "frame_num_"+str(frame_number)+".png"), image)

    def _run(self):
        plt.figure()
        for test_file in self.test_data_list:
            self.video_capture= self._read_vid_file(test_file)
            success = True
            frame_num = 0
            gt_arr = []
            while success:
                success, image = self.video_capture.read()
                if success is not True:
                    break
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self._resize_input_frame(image)
                data_input = self._box_roi(image, padding=self.padding)
                self._save_frame_img(data_input, test_file, frame_num)

                # cv2.imshow("data show", cv2.cvtColor(data_input, cv2.COLOR_RGB2BGR)/255)
                # cv2.moveWindow("data show", 0, 0)
                # print("input the ground truth of frame ", str(frame_num), ":   ")
                # if cv2.waitKey() == ord("1"):
                #     print("1 is inserted to array")
                #     gt_arr.append(int(1))
                # else:
                #     print("0 is inserted to array")
                #     gt_arr.append(int(0))

                # plt.imshow(data_input / 255.)
                # plt.axis("off")

                # gt_arr.append(int(input("input gt of frame number...  "+str(frame_num))))
                # print(str(gt_arr[-1])+" is saved of frame number "+str(frame_num))
                frame_num += 1

            # self.csv_write(gt_arr, test_file)
        return

    def _resize_input_frame(self, image):
        resized_img = cv2.resize(image, dsize=(720, 480), interpolation=cv2.INTER_LANCZOS4)
        return resized_img

    def _model_weight_loader(self):
        self.model.load_weights(self.weight_path)

    def _box_roi(self, input_frame, padding=0):
        height, width, ch = input_frame.shape
        input_frame = tf.keras.utils.array_to_img(input_frame)
        draw = ImageDraw.Draw(input_frame)
        draw.rectangle((padding, height * 2 // 3, width-padding, height), outline=(0, 255, 0), width=3)
        input_frame = tf.keras.utils.img_to_array(input_frame)
        return input_frame

    def _crop_roi(self, input_frame, padding = 0):
        height, width, ch = input_frame.shape
        cropped_img = input_frame[height * 2 // 3:, padding:width - padding, :]
        return cropped_img

    def _get_subtitle_mapping_result(self, data_input):
        data_input = unsqueeze(data_input/255.)
        output = self.model.predict(data_input, verbose = 2)
        return output, np.sum(output)

    def _subtitle_mapping_full_size_recon(self, input_frame, roi_mapping):
        original_height, original_width = self.frame_size
        cropped_height_start_y = original_height * 2 // 3
        recon_mapping = np.zeros(shape=(original_height, original_width, 1))
        recon_mapping[cropped_height_start_y:, self.padding:original_width - self.padding, :] = roi_mapping
        input_frame[:, :, 0:1] += (recon_mapping*255).astype(np.uint8)
        mapping_frame = np.clip(input_frame, 0, 255)
        # plt.imshow(mapping_frame)
        return recon_mapping, mapping_frame

    def _plot_on_screen(self, result, winname):
        cv2.imshow(winname, result)

    def _subtitle_prob_burn_in(self, image, subtitle_prob):
        image_pil = tf.keras.utils.array_to_img(image)
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype(font="./utils/movie_fonts/arial_bold.ttf", size=32)
        if subtitle_prob > 1500:
            draw.text((10, 10), str(subtitle_prob)+"Sentence Detected!!", fill="red", font=font)
        else:
            draw.text((10, 10), str(subtitle_prob)+"...", fill="blue", font=font)
        return tf.keras.utils.img_to_array(image_pil).astype(np.uint8)

    def _histogram_analysis_burn_in(self, image, histogram_count):
        image_pil = tf.keras.utils.array_to_img(image)
        if histogram_count > 10000:
            draw = ImageDraw.Draw(image_pil)
            font = ImageFont.truetype(font="./utils/movie_fonts/arial_bold.ttf", size=32)
            # draw.text((10, 40), str(histogram_count), fill="blue", font=font)
            draw.text((10, 40), str(histogram_count)+"White Background", fill="blue", font=font)
        else:
            draw = ImageDraw.Draw(image_pil)
            font = ImageFont.truetype(font="./utils/movie_fonts/arial_bold.ttf", size=32)
            draw.text((10, 40), str(histogram_count)+"...", fill="blue", font=font)
        # plt.figure()
        # plt.imshow(image_pil)
        return tf.keras.utils.img_to_array(image_pil).astype(np.uint8)

    def _save_in_dir(self, result):
        # print(self.video_writer.isOpened())
        # plt.imshow(result)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        self.video_writer.write(result)

    def _histogram_analyser(self, image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        histr = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist_count = int(np.sum(histr[200:]))
        return hist_count

    def _read_vid_file(self, data_file_name):
        data_path = os.path.join(self.data_dir, data_file_name)
        if os.path.isfile(data_path):
            print(data_path+"is loaded")
            vidcap = cv2.VideoCapture(data_path)
            return vidcap
        else:
            print("there is no file in this directory")
            return


# dataloader = CustomDataLoader("../datasets/train_dataset/indoor")
# x, y = dataloader.data_dir()

video_dir = "../datasets/test_dataset/video_source" #dataset_path
output_video_dir = "../datasets/test_dataset/classification_result"

test_data_loader = ReadVideoAsData(dataset_dir=video_dir, output_dir=output_video_dir)

test_data_loader._run()