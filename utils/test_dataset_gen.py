import cv2
import numpy as np
from utils.utilities import unsqueeze

def get_input_data(start_time, end_time, file_path = "/Users/iptvpeullaespomgaebaltim/Documents/pythoncode/subscription_finder/datasets/yesman.mp4"):
    vidcap = cv2.VideoCapture(file_path)
    vid_spf = 1/vidcap.get(cv2.CAP_PROP_FPS)
    frame_num = 0

    success, image = vidcap.read()
    time_stamp = 0
    test_dataset = np.array(unsqueeze(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    while success and time_stamp<end_time:
        frame_num += 1
        success, image = vidcap.read()
        time_stamp = frame_num * vid_spf
        if time_stamp > start_time:
            test_dataset = np.append(test_dataset, unsqueeze(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), axis=0)
    return test_dataset

