import tensorflow as tf
import os
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import pathlib

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, file_path):
        self.model = model
        self.file_path = file_path

    def _representative_data_gen(self):
        for _ in range(100):
            data = np.random.rand(1, 160, 440, 3)
            yield [data.astype(np.float32)]
        # for input_value in tf.data.Dataset.from_tensor_slices(self.data_sample).batch(1).take(100):
        #     yield [np.float32(input_value)]


    def on_epoch_end(self, epoch, logs=None):
        if not os.path.isdir(self.file_path):
            os.mkdir(self.file_path)
        self.model.save_weights(os.path.join(self.file_path, str(epoch) + ".h5"))
        self.model.save_weights(os.path.join(self.file_path, str(epoch) + ".tflite"))

        # converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        # self.model.build((1, 160, 440, 3))
        # converter.representative_dataset = self._representative_data_gen()
        # tflite_model = converter.convert()


        ###########################################################################
        # converter = tf.lite.TFLiteConverter.from_saved_model(self.file_path)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.representative_dataset = self._representative_data_gen()
        # converter.target_spec.supported_ops = [
        #     tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        #     tf.lite.OpsSet.SELECT_TF_OPS
        # ]
        #
        # converter.experimental_new_converter = True
        # converter.experimental_enable_resource_variables = True
        #
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8
        # tflite_quant_model = converter.convert()
        #
        # interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
        # input_type = interpreter.get_input_details()[0]['dtype']
        # print('input: ', input_type)
        # output_type = interpreter.get_output_details()[0]['dtype']
        # print('output: ', output_type)
        #
        # tflite_models_dir = pathlib.Path(self.file_path)
        # tflite_models_dir.mkdir(exist_ok=True, parents=True)
        #
        # tflite_model_quant_file = tflite_models_dir / "mnist_model_quant.tflite"
        # tflite_model_quant_file.write_bytes(tflite_quant_model)



    # def on_epoch_end(self, epoch, logs=None):
    #     if not os.path.isdir(self.file_path):
    #         os.mkdir(self.file_path)
    #     self.model.save_weights(os.path.join(self.file_path, str(epoch)+".h5"))
    #     self.model.save_weights(os.path.join(self.file_path, str(epoch)+".pb"))
    #     self.model.save_weights(os.path.join(self.file_path, str(epoch)+".pbtxt"))
    #     self.model.save(self.file_path)
    #     converter_quant = tf.lite.TFLiteConverter.from_keras_model(self.model)
    #     converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    #     converter_quant.representative_dataset = self._representative_data_gen
    #     converter_quant.target_spec.supported_ops = [
    #         tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    #         tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    #         tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    #     ]
    #     converter_quant.inference_input_type = tf.uint8
    #     converter_quant.inference_output_type = tf.uint8
    #     tflite_model_quant = converter_quant.convert()
    #     # with open(os.path.join(self.file_path, str(epoch)+"_uint8_quant.tflite"), "wb") as f:
    #     #     f.write(tflite_model_quant)
    #     interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    #     input_type = interpreter.get_input_details()[0]['dtype']
    #     print('input: ', input_type)
    #     output_type = interpreter.get_output_details()[0]['dtype']
    #     print('output: ', output_type)
    #
    #     tflite_models_dir = pathlib.Path(self.file_path)
    #     tflite_models_dir.mkdir(exist_ok=True, parents=True)
    #     tflite_model_quant_file = tflite_models_dir / str(epoch)+"_uint8_quant.tflite"
    #     tflite_model_quant_file.write_bytes(tflite_model_quant)
