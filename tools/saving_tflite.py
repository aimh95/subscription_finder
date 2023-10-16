import pathlib

from models.real_u_net import U_Net
from tensorflow.python.tools import freeze_graph
from data_gen import data_generation_test_ver
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from PIL import Image
from utils.custom_utils import unsqueeze
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib


weight_path = "/Users/pythoncodes/subtitle_finder/weight_path/u_net_total_to_custom_finetuning_epochs/110.h5"
video_path = "/Users/pythoncodes/subtitle_finder/datasets/test_dataset/video_source/ScaryMovie4.mp4"  #dataset_path
output_video_dir = "/Users/pythoncodes/subtitle_finder/temp"


model = U_Net()

# model.summary()
# img = unsqueeze(tf.keras.utils.img_to_array(Image.open("/Users/pythoncodes/subtitle_finder/tools/sample.jpeg").resize((720, 480)))/255)
# plt.imshow(model.predict(img)[0])



# # converter = tf.lite.TFLiteConverter.from_keras_model(model.build((1, 160, 440, 3)))
# converter = tf.lite.TFLiteConverter.from_saved_model(model.build((1, 160, 440, 3)))
# converter.convert()


# convert model to tflite and quantize to int8
# model.save_weights("/Users/pythoncodes/subtitle_finder/tools/test.pb")

# #keras 모델을 concrete function format으로 변환
def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 160, 440, 3)
      yield [data.astype(np.float32)]

model.build(input_shape=(1, 160, 440, 3))
model.load_weights(weight_path)

full_model = tf.function(lambda x:model(x))
full_model = full_model.get_concrete_function(tf.TensorSpec((1, 160, 440, 3)))
frozen_func = convert_variables_to_constants_v2(full_model)

export_dir = "/Users/pythoncodes/subtitle_finder/tools"
tf.saved_model.save(model, export_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)



tflite_model_file = pathlib.Path(os.path.join(export_dir, "quantized_model.tflite"))
tflite_model_file.write_bytes(tflite_model)

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_log",
                  name=f'test.pb',
                  as_text=False)

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_log",
                  name=f'test.pbtxt',
                  as_text=True)

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_log",
                  name=f'test.tflite',
                  as_text=True)

# converter = tf.lite.TFLiteConverter.from_saved_model("/Users/pythoncodes/subtitle_finder/tools")
