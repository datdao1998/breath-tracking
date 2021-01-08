import numpy as np
import keras
from scipy.io import wavfile
import librosa
import os


class BreathDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, directory,
                 list_labels=['normal', 'deep', 'rush', 'other'],
                 batch_size=32,
                 dim=None,
                 classes=None,
                 shuffle=True):
        'Initialization'
        self.directory = directory
        self.list_labels = list_labels
        self.dim = dim
        self.__flow_from_directory(self.directory)
        self.batch_size = batch_size
        self.classes = len(self.list_labels)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.wavs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        rawX = [self.wavs[k] for k in indexes]
        rawY = [self.labels[k] for k in indexes]

        # Generate data
        X, Y = self.__feature_extraction(rawX, rawY)

        return X, Y

    def __flow_from_directory(self, directory):
        self.wavs = []
        self.labels = []
        for dir in os.listdir(directory):
            sub_dir = os.path.join(directory, dir)
            if os.path.isdir(sub_dir) and dir in self.list_labels:
                label = self.list_labels.index(dir)
                for file in os.listdir(sub_dir):
                    self.wavs.append(os.path.join(sub_dir, file))
                    self.labels.append(label)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.wavs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __feature_extraction(self, list_wav, list_label):
        'Generates data containing batch_size samples'
        # Initialization
        X = []
        Y = []

        # Generate data
        for i in range(self.batch_size):
            rate, data = wavfile.read(list_wav[i])
            data = np.array(data, dtype=np.float32)
            if data.size > 64000:
                start = int((data.size - 64000) / 2)
                end = start + 64000
                data = data[start:end]

            if data.size < 64000:
                for pad in range(0, 64000 - data.size):
                    data = np.append(data, 0.0)

            data *= 1. / 32768
            feature = librosa.feature.mfcc(y=data, sr=rate,
                                           n_mfcc=40, fmin=0, fmax=8000,
                                           n_fft=int(16 * 64), hop_length=int(16 * 32), power=2.0)
            # print(feature.shape)
            feature = np.reshape(feature, self.dim)
            X.append(feature)
            Y.append(list_label[i])

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=int)
        return X, Y
