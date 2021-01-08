import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.engine import InputLayer
from keras import backend as K

class SimpleCNN(object):
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Conv2D(16, (8, 20), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, (4, 10), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # models.add(Conv2D(64, (4, 10), activation='relu'))
        # models.add(MaxPooling2D(pool_size=(2, 2)))
        # models.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        return model
