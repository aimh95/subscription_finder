from utils.test_dataset_gen import get_input_data, crop_img
from models.u_net_base import unet_base
from utils.test_dataset_gen import unsqueeze
import matplotlib.pyplot as plt
import os

model = unet_base()
test_dataset = get_input_data()

check_point_path = "./check_point/unet_basic/cp.ckpt"
check_point_dir = os.path.dirname(check_point_path)
model.load_weights(check_point_path)
plt.figure()
for i, image in enumerate(test_dataset):
    # plt.imshow(image)
    model_input = crop_img(image)/255.
    plt.imshow(model_input)
    y_pred = model.predict(unsqueeze(model_input))
    print(y_pred)
    if y_pred>0.5:
        pass
    pass

print(y_pred)