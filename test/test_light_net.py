from models.light_network import LightNetImageFeatureExtract, LightNetClassifier
from data_gen import data_generation_test_ver

padding = 100



video_path = "../datasets/test_dataset/video_source/ScaryMovie4.mp4"  #dataset_path
output_video_dir = "../results/"
# model_weight_path = "weight_path/real_u_net_light_totaltext_150epochs_load/cp-0141.ckpt"
model_weight_path = "weight_path/real_u_net_totaltext_100epochs/cp-0071.ckpt"

test_data_loader = data_generation_test_ver.ReadVideoAsData(dataset_path=video_path, output_path=output_video_dir, model_weight_path=model_weight_path, model=model)

test_data_loader._run()

#ndarray: 232 880 3 [[[0.17647059 0.