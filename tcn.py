from keras.models import Model
from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense


def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(
        x)  # first convolution
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)  # Second convolution
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  # sortcut (shortcut)
    o = add([r, shortcut])
    o = Activation('relu')(o)  # Activation function
    return o


class TCN(object):
    @staticmethod
    def build(input_shape, classes):
        inputs = Input(shape = input_shape)
        x = ResBlock(inputs, filters=32, kernel_size=3, dilation_rate=1)
        x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=2)
        x = ResBlock(x, filters=16, kernel_size=3, dilation_rate=4)
        x = Flatten()(x)
        x = Dense(classes, activation='softmax')(x)
        model = Model(input=inputs, output=x)
        return model


