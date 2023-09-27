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
from utils.utilities import unsqueeze

class ReadVideoAsData(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, output_path, model_weight_path, model, padding = 140):
        super(ReadVideoAsData, self).__init__()
        # self.dataset_path = dataset_path
        self.data_dir = dataset_path
        self.output_dir = output_path
        self.weight_path = model_weight_path

        self.model = model
        self.padding = padding

        self.video_capture, self.video_spf = self._read_vid_file()
        self.video_writer = self._video_writer_initializer()

    def _run(self, start_time=None, end_time=None, frame=None):
        self._model_weight_loader()
        success = True
        time_stamp = 0
        frame_num = 0

        while success:
            frame_num += 1
            success, image = self.video_capture.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self._resize_input_frame(image)
            time_stamp = frame_num * self.video_spf

            if start_time == None and end_time == None:
                data_input = self._crop_roi(image, padding = self.padding)
                histogram_count = self._histogram_analyser(data_input)
                mapping_result_of_roi, subtitle_prob = self._get_subtitle_mapping_result(data_input)
                mapping_result, mapping_result_on_frame = self._subtitle_mapping_full_size_recon(image, mapping_result_of_roi)
                mapping_result_on_frame = self._subtitle_prob_burn_in(mapping_result_on_frame, subtitle_prob)
                mapping_result_on_frame = self._histogram_analysis_burn_in(mapping_result_on_frame, histogram_count)
                self._save_in_dir(mapping_result_on_frame)

            elif start_time <= time_stamp and end_time >= time_stamp:
                data_input = self._crop_roi(image, padding=self.padding)
                histogram_count = self._histogram_analyser(data_input)
                mapping_result_of_roi, subtitle_prob = self._get_subtitle_mapping_result(data_input)
                mapping_result, mapping_result_on_frame = self._subtitle_mapping_full_size_recon(image, mapping_result_of_roi)
                mapping_result_on_frame = self._subtitle_prob_burn_in(mapping_result_on_frame, subtitle_prob)
                mapping_result_on_frame = self._histogram_analysis_burn_in(mapping_result_on_frame, histogram_count)
                self._save_in_dir(mapping_result_on_frame)

            if frame!=None and frame_num==frame:
                data_input = self._crop_roi(image, padding=self.padding)
                mapping_result_of_roi, subtitle_prob = self._get_subtitle_mapping_result(data_input)
                mapping_result, mapping_result_on_frame = self._subtitle_mapping_full_size_recon(image, mapping_result_of_roi)
                histogram_count = self._histogram_analyser(data_input)
                mapping_result_on_frame = self._subtitle_prob_burn_in(mapping_result_on_frame, subtitle_prob)
                mapping_result_on_frame = self._histogram_analysis_burn_in(mapping_result_on_frame, histogram_count)

                self._capture_spcf_frame(mapping_result_on_frame, frame_num)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return

    def _resize_input_frame(self, image):
        resized_img = cv2.resize(image, dsize=(720, 480), interpolation=cv2.INTER_LANCZOS4)
        return resized_img
    def _capture_spcf_frame(self, image, frame):
        output_dir = os.path.join(self.output_dir, self.weight_path.split("/")[1], str(frame))
        cv2.imwrite(output_dir, image)
        return

    def _model_weight_loader(self):
        self.model.load_weights(self.weight_path)

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

    def _video_writer_initializer(self):
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        output_dir = os.path.join(self.output_dir, self.weight_path.split("/")[1])
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        file_name = self.weight_path.split(".")[0][-4:]+self.data_dir.split("/")[-1]
        mapping_video_writer = cv2.VideoWriter(os.path.join(output_dir, file_name), fourcc, 1/self.video_spf, (self.frame_size[1], self.frame_size[0]))
        # concat_output = cv2.VideoWriter('concat_output.mp4', fourcc, 23.976, (1280, 694 * 2))
        return mapping_video_writer

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

    def _read_vid_file(self):
        if os.path.isfile(self.data_dir):
            vidcap = cv2.VideoCapture(self.data_dir)
            # vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
            # vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
            vid_spf = 1 / vidcap.get(cv2.CAP_PROP_FPS)
            # self.frame_size = (int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            self.frame_size = (480, 720)
            return vidcap, vid_spf
        else:
            print("there is no file in this directory")
            return


# dataloader = CustomDataLoader("../datasets/train_dataset/indoor")
# x, y = dataloader.data_dir()