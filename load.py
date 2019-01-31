import os
import datetime
import json
from collections import OrderedDict

from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

def load_mobilenet(image_class):

    root_path = '/home/yu/Documents/MURA/'
    valid_path = root_path + 'v_valid/{}'
    valid_path = valid_path.format(image_class)

    idg_valid_settings = dict(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=0,
        width_shift_range=0.,
        height_shift_range=0.,
        zoom_range=0.0,
        horizontal_flip=False,
        vertical_flip=False)
    idg_valid = ImageDataGenerator(**idg_valid_settings)

    valid_gen = idg_valid.flow_from_directory(
        valid_path,
        follow_links=True,
        target_size=(224, 224),
        color_mode='grayscale')

    weights_path = "/home/yu/Documents/MobileNet/MobileNetV2/weights/XR_HUMERUS_MobileNet_01_25_20_06_weights.hdf5"
    model = MobileNetV2(classes=2 , weights=weights_path, input_shape=(224, 224, 1))
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print('Layers: {}, parameters: {}'.format(len(model.layers), model.count_params()))

    scores = model.evaluate_generator(valid_gen, steps=len(valid_gen), verbose=1)
    print("Accuracy:%s" % scores[1])

if __name__=='__main__':
    load_mobilenet('XR_HUMERUS')
    # load_mobilenet('XR_WRIST')