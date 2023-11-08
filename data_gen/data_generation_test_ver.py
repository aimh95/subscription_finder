from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import cv2
import os
import numpy as np
from utils.custom_utils import unsqueeze
from models.real_u_net import EncoderConvBlock, DecoderConvBlock, U_Net
import matplotlib.pyplot as plt
from utils.custom_utils import grad
from tensorflow.keras.metrics import binary_accuracy, Precision, Recall
import csv

from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
class ReadVideoAsData(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, output_path, model_weight_path, model, padding = 140, classification_mode=0):
        super(ReadVideoAsData, self).__init__()
        # self.dataset_path = dataset_path
        self.data_dir = dataset_path
        self.output_dir = output_path
        self.weight_path = model_weight_path

        self.model = model
        self.padding = padding
        self.classification_mode = classification_mode

        self.video_capture, self.video_spf = self._read_vid_file()
        self.video_writer = self._video_writer_initializer()
        self.ground_truth = self._get_ground_truth()
        self.classification_result = []

    def _get_ground_truth(self):
        csv_name = self.data_dir.split("/")[-1].split(".")[0]+".csv"
        gt_dir = "../datasets/test_dataset/classification_result"

        ground_truth = []
        with open(os.path.join(gt_dir, csv_name), 'r', encoding="utf-8") as f:
            next(f)
            csv_reader = csv.reader(f)
            for line in csv_reader:
                ground_truth.append(int(line[-1]))
        return ground_truth

    def _run(self, start_time=None, end_time=None, frame=None):
        success = True
        time_stamp = 0
        frame_num = 0

        while success:
            frame_num += 1
            success, image = self.video_capture.read()
            if not success:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self._resize_input_frame(image)
            time_stamp = frame_num * self.video_spf
            data_input = self._crop_roi(image, padding=self.padding)
            if frame_num==1:
                self._model_weight_loader(data_input)

            if start_time == None and end_time == None:
                histogram_count = self._histogram_analyser(data_input)
                mapping_result_of_roi, subtitle_prob, maximum_heatmap = self._get_subtitle_mapping_result(data_input)
                self.classification_result.append(1) if (maximum_heatmap > 0.999 and subtitle_prob>800) else self.classification_result.append(0)
                mapping_result, mapping_result_on_frame = self._subtitle_mapping_full_size_recon(image,mapping_result_of_roi)
                mapping_result_on_frame = self._subtitle_prob_burn_in(mapping_result_on_frame, subtitle_prob, maximum_heatmap)
                mapping_result_on_frame = self._histogram_analysis_burn_in(mapping_result_on_frame, histogram_count)
                self._save_in_dir(mapping_result_on_frame)
                cv2.imshow("result", cv2.cvtColor(mapping_result_on_frame, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)

            elif start_time <= time_stamp and end_time >= time_stamp:
                data_input = self._crop_roi(image, padding=self.padding)
                histogram_count = self._histogram_analyser(data_input)
                mapping_result_of_roi, subtitle_prob, maximum_heatmap = self._get_subtitle_mapping_result(data_input)
                self.classification_result.append(1) if subtitle_prob > 1500 else self.classification_result.append(0)
                mapping_result, mapping_result_on_frame = self._subtitle_mapping_full_size_recon(image, mapping_result_of_roi)
                mapping_result_on_frame = self._subtitle_prob_burn_in(mapping_result_on_frame, subtitle_prob, maximum_heatmap)
                mapping_result_on_frame = self._histogram_analysis_burn_in(mapping_result_on_frame, histogram_count)
                self._save_in_dir(mapping_result_on_frame)
                cv2.imshow("result", cv2.cvtColor(mapping_result_on_frame, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)
        if start_time != None or end_time != None:
            self.ground_truth = self.ground_truth[int(start_time // self.video_spf):int(end_time // self.video_spf)]
        accuracy,precision, recall, f1_score = self._get_classification_accuracy()
        print("accuracy: ", accuracy)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1_score: ", f1_score)

    def _get_classification_accuracy(self):
        precision_calc = Precision()
        recall_calc = Recall()

        precision_calc.update_state(self.ground_truth, self.classification_result)
        recall_calc.update_state(self.ground_truth, self.classification_result)

        accuracy = binary_accuracy(self.ground_truth, self.classification_result)
        precision = precision_calc.result().numpy()
        recall = recall_calc.result().numpy()

        return accuracy, precision, recall



    def _feature_map_visualization(self, input_img):
        vis_model = []
        feature_map = []
        input_img = tf.expand_dims(input_img/255., axis=0)
        feature_map.append(input_img)
        plt.figure()
        plt.imshow(input_img[0])
        plt.show

        for i in range(len(self.model.layers)):
            layer = self.model.layers[i]
            if 'encoder' in layer.name:
                plt.figure()
                weight_loaded = layer.get_weights()
                temp_model = EncoderConvBlock(filters=32 * 2**(len(feature_map)-1))
                temp_model.build(input_shape=(feature_map[-1].shape))
                temp_model.set_weights(weight_loaded)
                prediction = temp_model.predict(feature_map[-1])
                for i in range(prediction.shape[3]):
                    plt.subplot(4 * 2**(len(feature_map)-1), 8, i+1)
                    plt.imshow(prediction[0,:,:,i])
                    plt.axis("off")
                    plt.show()
                feature_map.append(prediction)
            if 'decoder' in layer.name:
                plt.figure()
                weight_loaded = layer.get_weights()
                temp_model = DecoderConvBlock(filters=32*2**(6-len(feature_map)))
                temp_model.build(input_shape=(feature_map[-1].shape))
                temp_model.set_weights(weight_loaded)
                prediction = temp_model.predict(feature_map[-1])
                for i in range(prediction.shape[3]):
                    plt.subplot(4*2**(6-len(feature_map)), 8, i+1 )
                    plt.imshow(prediction[0,:,:,i])
                    plt.axis("off")
                    plt.show()
                feature_map.append(prediction)

    def _resize_input_frame(self, image):
        resized_img = cv2.resize(image, dsize=(720, 480), interpolation=cv2.INTER_LANCZOS4)
        return resized_img

    def _capture_spcf_frame(self, image, frame):
        output_dir = os.path.join(self.output_dir, self.weight_path.split("/")[1], str(frame))
        cv2.imwrite(output_dir, image)
        return

    def _model_weight_loader(self, data_input):
        self.model.build(unsqueeze(data_input).shape)
        print("model is loaded from "+self.weight_path)
        self.model.load_weights(self.weight_path)

    def _crop_roi(self, input_frame, padding = 0):
        height, width, ch = input_frame.shape
        cropped_img = input_frame[height * 2 // 3:, padding:width - padding, :]
        return cropped_img

    def _get_subtitle_mapping_result(self, data_input):
        data_input = unsqueeze(data_input/255.)
        output = self.model.predict(data_input, verbose = 2)
        return output, np.sum(output), np.max(output)

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

    def _subtitle_prob_burn_in(self, image, subtitle_prob, subtitle_max):
        image_pil = array_to_img(image)
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype(font="../utils/movie_fonts/arial_bold.ttf", size=32)
        if subtitle_prob > 1500:
            draw.text((10, 10), str(subtitle_prob)+"|"+str(subtitle_max)+"Sentence Detected!!", fill="red", font=font)
        else:
            draw.text((10, 10), str(subtitle_prob)+"|"+str(subtitle_max)+"...", fill="blue", font=font)
        return img_to_array(image_pil).astype(np.uint8)

    def _histogram_analysis_burn_in(self, image, histogram_count):
        image_pil = array_to_img(image)
        if histogram_count > 10000:
            draw = ImageDraw.Draw(image_pil)
            font = ImageFont.truetype(font="../utils/movie_fonts/arial_bold.ttf", size=32)
            # draw.text((10, 40), str(histogram_count), fill="blue", font=font)
            draw.text((10, 40), str(histogram_count)+"White Background", fill="blue", font=font)
        else:
            draw = ImageDraw.Draw(image_pil)
            font = ImageFont.truetype(font="../utils/movie_fonts/arial_bold.ttf", size=32)
            draw.text((10, 40), str(histogram_count)+"...", fill="blue", font=font)
        # plt.figure()
        # plt.imshow(image_pil)
        return img_to_array(image_pil).astype(np.uint8)

    def _video_writer_initializer(self):
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        output_dir = os.path.join(self.output_dir, self.weight_path.split("/")[-2])
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