import os
import datetime
import json
from collections import OrderedDict

from keras.applications.mobilenetv2 import MobileNetV2
# from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model


# from mobilenet_v2 import MobileNetv2                          # use different models
# from mobilenet_v2_self_design import build_mobilenet_v2

def log_mobilenet(mkfile_time, image, time, loss, acc, val_loss, val_acc, layers, parameters, steps):
    if not os.path.exists('log'):
        os.mkdir('log')
    mkfile_name = image + '_MobileNet_V2_' + mkfile_time + '.json'
    json_file = os.path.join('./log', mkfile_name)  # use time to name the log file
    with open(json_file, 'a') as fp:
        d_log = {}
        d_log['Model'] = 'MobileNet_V2'
        d_log['Class'] = image
        d_log['Time for training'] = time
        d_log['Loss'] = loss
        d_log['Accuracy'] = acc
        d_log['Loss_valid'] = val_loss
        d_log['Acc_valid'] = val_acc
        d_log['Layers'] = layers
        d_log['Parameters'] = parameters
        d_log['Steps'] = steps
        # json.dump(d_log, fp, indent=4, sort_keys=True, default=str)
        json.dump(OrderedDict(d_log), fp, indent=4, default=str)


def train_mobilenet(mkfile_time, image_class, epochs, steps):
    start_train_time = datetime.datetime.now()

    root_path = '/home/yu/Documents/MURA/'
    train_path = root_path + 'v_train/{}'
    valid_path = root_path + 'v_valid/{}'

    train_path = train_path.format(image_class)
    valid_path = valid_path.format(image_class)

    idg_train_settings = dict(  # Data preprocessing
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True)
    idg_train = ImageDataGenerator(**idg_train_settings)

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

    train_gen = idg_train.flow_from_directory(
        train_path,
        follow_links=True,
        batch_size=64,
        target_size=(224, 224),
        color_mode='grayscale')

    valid_gen = idg_valid.flow_from_directory(
        valid_path,
        follow_links=True,
        batch_size=64,
        target_size=(224, 224),
        color_mode='grayscale')

    # use different models to try

    a, b = next(valid_gen)
    model = MobileNetV2(classes=b.shape[1], weights=None, input_shape=a.shape[1:])
    # model = MobileNet(classes=b.shape[1], weights=None, input_shape=a.shape[1:], dropout=0.5)
    # model = MobileNetv2((224, 224, 1), 2)
    # model = build_mobilenet_v2((224, 224, 1))
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print('Layers: {}, parameters: {}'.format(len(model.layers), model.count_params()))

    if not os.path.exists('weights'):
        os.mkdir('weights/')
    file_path = os.path.join('weights/', image_class)
    file_path = file_path + '_MobileNet_V2_' + mkfile_time + '_weights.hdf5'

    checkpoint = ModelCheckpoint(
        file_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    early = EarlyStopping(monitor="val_acc",
                          mode="max",
                          patience=3)
    callbacks_list = [checkpoint, early]  # early

    hist = model.fit_generator(
        train_gen,
        steps_per_epoch=steps,  # default 30
        validation_data=valid_gen,
        validation_steps=10,
        epochs=epochs,
        callbacks=callbacks_list)
    print(hist.history)

    end_train_time = datetime.datetime.now()
    time_train = end_train_time - start_train_time

    loss = hist.history['loss']
    acc = hist.history['acc']
    val_loss = hist.history['val_loss']
    val_acc = hist.history['val_acc']
    layers = len(model.layers)
    parameters = model.count_params()

    # log
    log_mobilenet(mkfile_time, image_class, time_train, loss, acc, val_loss, val_acc, layers, parameters, steps)

    return time_train


if __name__ == '__main__':
    # data = ['XR_ELBOW',
    #         'XR_FINGER',
    #         'XR_FOREARM',
    #         'XR_HAND',
    #         'XR_HUMERUS',
    #         'XR_SHOULDER',
    #         'XR_WRIST']
    # data = ['XR_WRIST', 'XR_HAND']
    data = ['XR_HUMERUS']
    time_all = []
    loss_all = []
    acc_all = []
    steps = [125]
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%m_%d_%H_%M')
    for i, image_class in enumerate(data):
        for j, steps_per_epoch in enumerate(steps):
            time = train_mobilenet(mkfile_time, image_class, epochs=30, steps=steps_per_epoch)
            time_all.append(time)
            print('Total training time: %s' % time_all[j])




