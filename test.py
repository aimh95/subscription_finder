from utils.test_dataset_gen import get_input_data
from models.u_net_base import unet_base
import os

model = unet_base()
input_img = get_input_data()

check_point_path = "./check_point/unet_basic/cp.ckpt"
check_point_dir = os.path.dirname(check_point_path)

model.load_weights(check_point_path)

y_pred = model.predict(input_img)
print(y_pred)