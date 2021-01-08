#!/usr/bin/env python
# coding: utf-8
import itertools

import keras
import numpy as np
import os
import pickle

import seaborn
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import librosa
from dataset import BreathDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import scikitplot as skplt

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

BATCH_SIZE = 32
LIST_LABELS = ['normal', 'deep', 'strong','other']
N_CLASSES = len(LIST_LABELS)
LR = 3
N_EPOCHS = 30
#INPUT_SIZE = (40, 126, 1)


INPUT_SIZE = (40, 126)

def main():
    train_generator = BreathDataGenerator(
        'E:/data_augmentation/training',
        list_labels=LIST_LABELS,
        batch_size=BATCH_SIZE,
        dim=INPUT_SIZE,
        shuffle=True)
    N_TRAIN_SAMPLES = len(train_generator.wavs)
    # train_generator.__getitem__(0)
    print("Train samples: {}".format(N_TRAIN_SAMPLES))
    # exit(1)
    validation_generator = BreathDataGenerator(
        'E:/data_augmentation/testing',
        list_labels=LIST_LABELS,
        batch_size=BATCH_SIZE,
        dim=INPUT_SIZE,
        shuffle=False)
    N_VALID_SAMPLES = len(validation_generator.wavs)
    print("Validation samples: {}".format(N_VALID_SAMPLES))

    # import keras.applications
    from keras.applications.mobilenet import MobileNet
    from keras.applications.mobilenet_v2 import MobileNetV2
    from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
    from keras.models import Sequential, Model
    from resnet import ResnetBuilder
    from cnn import SimpleCNN
    from lstm import SimpleLSTM
    from tcn import TCN

    models = TCN.build(input_shape=(INPUT_SIZE[0],INPUT_SIZE[1]),classes=N_CLASSES)
    #models = MobileNet(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1],1), include_top=True, classes=N_CLASSES, weights=None)
    # models = MobileNetV2(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1), include_top=True, classes=N_CLASSES,
    #                      weights=None)
    # models = ResnetBuilder.build_resnet_152(input_shape=INPUT_SIZE, num_outputs=N_CLASSES)
    # models = SimpleCNN.build(input_shape=INPUT_SIZE, classes=N_CLASSES)
    # models = SimpleLSTM.build(input_shape=INPUT_SIZE,classes=N_CLASSES)
    # model.summary()

    models.summary()

    # Training
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from keras.optimizers import Adadelta

    model_file = "final/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

    checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # tbCallBack = keras.callbacks.TensorBoard(log_dir='logs/tensorboard', write_graph=True, write_images=True)
    early_stoping = EarlyStopping(patience=5, monitor='val_loss')
    # callbacks_list = [checkpoint, tbCallBack]
    callbacks_list = [checkpoint, early_stoping]
    # models.load_weights("models/weights-improvement-02-0.39.hdf5")
    models.compile(loss=keras.losses.sparse_categorical_crossentropy,
                   optimizer=Adadelta(),
                   metrics=['accuracy'])

    mode = 'TRAIN'
    #mode = 'TEST'

    if mode == 'TRAIN':
        models.fit_generator(
            train_generator,
            steps_per_epoch=N_TRAIN_SAMPLES // BATCH_SIZE,
            initial_epoch=0,
            epochs=N_EPOCHS,
            validation_data=validation_generator,
            validation_steps=N_VALID_SAMPLES // BATCH_SIZE,
            callbacks=callbacks_list,
            max_queue_size=6,
            workers=3,
            use_multiprocessing=True,
        )

    else:
        best_model_path = "final/weights-improvement-09-0.00.hdf5"

        models.load_weights(best_model_path)

        test_generator = BreathDataGenerator(
            'E:/data_augmentation/testing',
            list_labels=LIST_LABELS,
            batch_size=1,
            dim=INPUT_SIZE,
            shuffle=False)
        N_TEST_SAMPLES = len(test_generator.wavs)
        print("Test samples: {}".format(N_TEST_SAMPLES))

        Y_pred = models.predict_generator(test_generator, N_TEST_SAMPLES)
        y_pred = np.argmax(Y_pred, axis=1)
        for i,y in enumerate(y_pred):
            if y==3 and test_generator.labels[i] == 0:
                print(test_generator.wavs[i])
        # print(y_pred)
        # print(test_generator.labels)
        # Plot confusion matrix
        cnf_matrix = confusion_matrix(test_generator.labels,y_pred)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes= LIST_LABELS, normalize=True,
                              title='Normalized confusion matrix')
        plt.savefig('report/fig1_mobile_net_2.png')
        plt.show()

        #console
        print('Confusion Matrix')
        print(confusion_matrix(test_generator.labels, y_pred))
        print('Classification Report')
        print(classification_report(test_generator.labels, y_pred, target_names=LIST_LABELS))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    main()
