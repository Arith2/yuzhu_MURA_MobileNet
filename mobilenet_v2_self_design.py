from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Activation, BatchNormalization, add, Reshape
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD


def build_mobilenet_v2(input_shape):

    model = Sequential()
    inputs = Input(input_shape)

    #   1st
    model.add(Conv2D(32, (2, 2), strides=(2, 2), input_shape=input_shape))
    # model.add(Conv2D(32, (3, 3), strides=(2, 2))(inputs))
    # model = Conv2D(32, (3, 3), strides=(2, 2))(inputs)
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    #   2nd
    model.add(Conv2D(32, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(16, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))

    #   3rd
    model.add(Conv2D(96, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(2, 2), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(24, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))
    # repeat 2 times
    model.add(Conv2D(144, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(24, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))

    # 4th
    model.add(Conv2D(144, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(2, 2), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(32, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))
    # repeat 3 times
    model.add(Conv2D(196, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(32, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))
    #
    model.add(Conv2D(196, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(32, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))

    # 5th
    model.add(Conv2D(196, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(2, 2), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(64, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))
    # repeat 4 times
    model.add(Conv2D(384, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(64, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))
    #
    model.add(Conv2D(384, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(64, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))
    #
    model.add(Conv2D(384, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(64, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))

    # 6th
    model.add(Conv2D(384, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(96, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))
    # repeat 3 times
    model.add(Conv2D(576, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(96, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))
    #
    model.add(Conv2D(576, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(96, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))

    # 7th
    model.add(Conv2D(576, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(2, 2), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(160, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))
    # repeat 3 times
    model.add(Conv2D(960, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(160, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))
    #
    model.add(Conv2D(960, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(160, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))

    # 8th
    model.add(Conv2D(960, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    model.add(Conv2D(320, (1, 1), strides=(1, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))

    # 9th
    model.add(Conv2D(1280, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(relu6))

    # 10th
    # use two FC to solve the problem of nonlinearity
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation(relu6))
#    model.add(Dense(2), activation='softmax')
    model.add(Dense(2))
    model.add(Activation('softmax'))
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


if __name__=='__main__':
    input_shape = (224, 224, 3)
    model = build_mobilenet_v2(input_shape)
    plot_model(model, to_file='images/MobileNetv2_myjob.png', show_shapes=True)
    model.summary()
