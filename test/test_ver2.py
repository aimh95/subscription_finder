from utils.test_dataset_gen import get_input_data
from utils.custom_utils import crop_img, numpyIMG_resize, cc_map_postprocessing, min_max_norm, sobel_operation
# from models.light_u_net import Light_U_Net
from models.real_u_net import U_Net
# from models.u_net_cbam import U_Net_CBAM
from data_gen import data_generation_test_ver

padding = 100


model = U_Net()
# model = U_Net_CBAM()

video_path = "./datasets/test_dataset/video_source/starwars.mp4"  #dataset_path
output_video_dir = "./results/"
# model_weight_path = "weight_path/real_u_net_light_totaltext_150epochs_load/cp-0141.ckpt"
# model_weight_path = "./weight_path/u_net_cbam_totaltext_100_epochs/149.h5"
# model_weight_path = "./weight_path/u_net_cbam_totaltext_200_epochs/599.h5"
model_weight_path = "./weight_path/real_u_net_totaltext_100epochs/cp-0001.ckpt"

test_data_loader = data_generation_test_ver.ReadVideoAsData(dataset_path=video_path, output_path=output_video_dir, model_weight_path=model_weight_path, model=model)

test_data_loader._run()

#ndarray: 232 880 3
# [[[0.17647059 0.12156863 0.06666667], [0.17647059 0.12156863 0.06666667], [0.17647059 0.12156863 0.06666667], ..., [0.13333333 0.08627451 0.0745098 ], [0.13333333 0.08627451 0.0745098 ],
# [0.13333333 0.08627451 0.0745098 ]],,
# [[0.17254902 0.11764706 0