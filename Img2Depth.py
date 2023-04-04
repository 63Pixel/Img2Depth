import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from tqdm import tqdm

from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.backend as K

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        if data_format is None:
            data_format = K.image_data_format()
        self.data_format = data_format
        self.size = size
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None

        return tf.image.resize(inputs, (height, width), method=tf.image.ResizeMethod.BILINEAR)

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def load_dense_depth_model(model_path):
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D}
    model = load_model(model_path, compile=False, custom_objects=custom_objects)
    return model


def image_to_depth_map(image, model):
    # Preprocess image
    input_image = cv2.resize(image, (640, 480))
    input_image = input_image.astype(np.float32) / 255.0
    input_batch = np.expand_dims(input_image, axis=0)

    # Calculate depth map
    depth_map = model.predict(input_batch)[0, :, :, 0]
    depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))

    return depth_map

def main():
    # Load model
    model_path = "dense_depth_model/nyu.h5"
    model = load_dense_depth_model(model_path)

    # Path of the directory where the script is located
    script_directory = os.path.dirname(os.path.realpath(__file__))

    # List of supported image formats
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    # Collect all image files in the directory
    image_files = [filename for filename in os.listdir(script_directory) if os.path.splitext(filename)[-1].lower() in image_extensions]

    # Iterate through all image files with a progress bar
    for filename in tqdm(image_files, desc="Processing images", unit="image"):
        # Load the image
        image = cv2.imread(os.path.join(script_directory, filename))

        # Create depth map
        depth_map = image_to_depth_map(image, model)

        # Save depth map
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        output_filename = os.path.splitext(filename)[0] + "_depth_map.png"
        cv2.imwrite(os.path.join(script_directory, output_filename), depth_map_normalized)

if __name__ == "__main__":
    main()