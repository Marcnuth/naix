from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from pathlib import Path
import shutil
import time
import numpy as np
import tensorflow as tf
from skimage import transform, color
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from naix.settings import DIR_MODELS


class WhiteScreenDetector:
    def __init__(self, model_file=DIR_MODELS / 'white_screen_cnn_1539251936.h5'):
        self.model = load_model(Path(model_file).absolute().as_posix())
        self.classes = ['normal', 'blank']

    def predict_file(self, image_file):
        img = load_img(Path(image_file).absolute().as_posix(), target_size=(96, 54))
        img_tensor = np.expand_dims(img_to_array(img), axis=0) / 255.0

        probs = self.model.predict(img_tensor, verbose=2)
        index = probs.argmax(axis=-1)[0]
        clz = self.classes[index]
        return clz, probs.max()

    def predict(self, image):
        img = transform.resize(color.gray2rgb(image), (96, 54), preserve_range=True)
        img_tensor = np.expand_dims(img_to_array(img), axis=0) / 255.0

        probs = self.model.predict(img_tensor, verbose=2)
        index = probs.argmax(axis=-1)[0]
        clz = self.classes[index]
        return clz, probs.max()

    def is_bug(self, image):
        return self.predict(image)[0] == 'blank'