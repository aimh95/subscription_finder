from utils.test_dataset_gen import get_input_data
from utils.utilities import crop_img, numpyIMG_resize, cc_map_postprocessing, min_max_norm, sobel_operation
from models.real_u_net_relu import U_Net
from utils.test_dataset_gen import unsqueeze
import matplotlib.pyplot as plt
import os
import time
import cv2
import numpy as np

padding = 100

model = U_Net()
test_dataset = get_input_data(25, 30, file_path = "./datasets/yesman.mp4")

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('prob_map.mp4', fourcc, 23.976, (1280, 694))
concat_output = cv2.VideoWriter('concat_output.mp4', fourcc, 23.976, (1280, 694*2))

check_point_path = "./check_point/real_u_net_datagen_point_spread/cp.ckpt"
check_point_dir = os.path.dirname(check_point_path)
model.load_weights(check_point_path)
for i, image in enumerate(test_dataset):
    # plt.imshow(image)
    model_input = crop_img(image, padding=padding)/255.
    # model_input, original_height, original_width = numpyIMG_resize(model_input, resize_shape=(320, 320))

    y_pred = model.predict(unsqueeze(model_input))
    y_pred_full = min_max_norm(cc_map_postprocessing(y_pred[0], padding = padding, original_size=image.shape[:2]))

    # cv2.imshow("input_img", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) #cv2.cvtColor(model_input, cv2.COLOR_RGB2BGR)
    # cv2.imshow("result",y_pred_full)
    output_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    sobel_operation(model_input)
    output_img[:, :, 2:3] += y_pred_full
    # cv2.imshow("result",output_img)
    concat_img = np.vstack((cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.cvtColor(y_pred_full, cv2.COLOR_GRAY2BGR)))
    concat_output.write(concat_img)
    cv2.imshow("pred_result", y_pred_full*256)
    cv2.imshow("full_img", concat_img)
    out.write(output_img)
    cv2.imshow("img",output_img)
    cv2.waitKey(1)
    pass
#ndarray: 232 880 3 [[[0.17647059 0.12156863 0.06666667],  [0.17647059 0.12156863 0.06666667],  [0.17647059 0.12156863 0.06666667],  ...,  [0.13333333 0.08627451 0.0745098 ],  [0.13333333 0.08627451 0.0745098 ],  [0.13333333 0.08627451 0.0745098 ]],, [[0.17254902 0.11764706 0