from pydub import AudioSegment
from keras.applications.mobilenet_v2 import MobileNetV2
from dataset import BreathDataGenerator
import numpy as np
import os, shutil
import streamlit as st
import pandas as pd

# Input audio file to be sliced

LIST_LABELS = ['normal', 'deep', 'strong', 'other']
N_CLASSES = len(LIST_LABELS)
INPUT_SIZE = (40, 126, 1)


def split_file(file, dir_file):
    # if file contain files
    for filename in os.listdir(dir_file):
        file_path = os.path.join(dir_file, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Get chunk files
    audio = AudioSegment.from_wav(file)

    # wav_dir = 'deep/'
    wav_dir = dir_file
    # Length of the audiofile in milliseconds
    n = len(audio)

    # Variable to count the number of sliced chunks
    counter = 1

    interval = 4 * 1000

    # Length of audio to overlap.
    # If length is 22 seconds, and interval is 5 seconds,
    # With overlap as 1.5 seconds,
    # The chunks created will be:
    # chunk1 : 0 - 5 seconds
    # chunk2 : 3.5 - 8.5 seconds
    # chunk3 : 7 - 12 seconds
    # chunk4 : 10.5 - 15.5 seconds
    # chunk5 : 14 - 19.5 seconds
    # chunk6 : 18 - 22 seconds
    overlap = 3.5 * 1000

    # Initialize start and end seconds to 0
    start = 0
    end = 0

    # Flag to keep track of end of file.
    # When audio reaches its end, flag is set to 1 and we break
    flag = 0

    # Iterate from 0 to end of the file,
    # with increment = interval
    for i in range(0, 2 * n, 500):

        # During first iteration,
        # start is 0, end is the interval
        if i == 0:
            start = 0
            end = interval

            # All other iterations,
        # start is the previous end - overlap
        # end becomes end + interval
        else:
            start = end - overlap
            end = start + interval

            # When end becomes greater than the file length,
        # end is set to the file length
        # flag is set to 1 to indicate break.
        if end >= n:
            end = n
            flag = 1

        # Storing audio file from the defined start to end
        chunk = audio[start:end]

        # Filename / Path to store the sliced audio
        filename = wav_dir + 'chunk' + str(counter) + '_' + str(int(start)) + '_' + str(int(end)) + '.wav'

        # Store the sliced audio file to the defined path
        chunk.export(filename, format="wav")
        # Print information about the current chunk
        # print("Processing chunk " + str(counter) + ". Start = "
        #       + str(start) + " end = " + str(end))

        # Increment counter for the next chunk
        counter = counter + 1

        if flag == 1:
            break


def load_data(chunk_files):
    return BreathDataGenerator(
        chunk_files,
        list_labels=LIST_LABELS,
        batch_size=1,
        dim=INPUT_SIZE,
        shuffle=False)


def classify(chunk_files, model_path):
    models = MobileNetV2(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1), include_top=True, classes=N_CLASSES,
                         weights=None)
    models.load_weights(model_path)
    predictors = load_data(chunk_files)
    N_TEST_SAMPLES = len(predictors.wavs)
    Y_pred = models.predict_generator(predictors, N_TEST_SAMPLES)
    y_pred = np.argmax(Y_pred, axis=1)
    return get_classify_data(y_pred)


def get_classify_data(y_pre):
    app = []
    result = []
    y_pre = y_pre[:-1]
    score = [[] for i in range(0, len(y_pre) + 8)]
    for i in range(0, len(y_pre)):
        for j in range(i, i + 8):
            score[j].append(y_pre[i])
    for i in range(0, len(y_pre)):
        app.append(max(set(score[i]), key=score[i].count))
    start = end = 0
    for i in range(0, len(y_pre) - 1):
        if not app[i] == app[i + 1]:
            end = i + 1
            label = app[i]
            points = [start * 0.5, end * 0.5, label]
            result.append(points)
            start = i + 1
    result.append([start * 0.5, len(y_pre) * 0.5 + 3.5, app[start]])
    return result


classify('overlapped_files','pretrained_model/model.hdf5')

def plot_point(plt, records, num_view, col):
    colors = ['g', 'r', 'blue', 'black']
    labels = ['normal', 'deep', 'strong', 'other']

    # initial plot
    plt.ylabel('Labels')
    plt.xlabel('Timestamps')
    plt.xticks(np.arange(0, 1000, 0.5))
    plt.yticks(np.arange(0, 4, 1))

    # plot figure
    color_points = []
    label_points = []
    for record in records:
        color_points.append(record[0])
        label_points.append(record[2])
    points = []
    for i in range(0, len(records)):
        points.append([color_points[i], label_points[i]])
    label_legend = []
    x_last = y_last = 0
    for i in range(0, num_view-1):
        x1 = points[i][0]
        y1 = points[i][1]

        x2 = points[i + 1][0]
        y2 = points[i + 1][1]
        x_last = x2
        y_last = y2
        if y1 not in label_legend:
            plt.plot([x1, x2], [y1, y2], colors[y1], label=labels[y1])
            label_legend.append(y1)
        else:
            plt.plot([x1, x2], [y1, y2], colors[y1])
    plt.plot([x_last,records[-1][1]],[y_last,y_last],colors[y_last])
    plt.plot()

    plt.legend()
    col.pyplot(plt)


def get_report(records, num_view, col, username):
    labels = ['normal', 'deep', 'strong', 'other']
    ###push log here
    ##push_log(records)
    records = records[:num_view]
    col.write('**Timestamp Detail**')
    col.write('**User: ' + username + '**')
    columns = ["Start time (s)", "End time (s)", "Label"]
    df_data = []
    for i, rd in enumerate(records):
        df_data.append([records[i][0], records[i][1], labels[records[i][2]]])
    df = pd.DataFrame(df_data, columns=columns)
    col.table(df)


def get_extra(records):
    total_duration = records[-1][1]
    duration = [0, 0, 0, 0]
    for r in records:
        duration[r[2]] += r[1] - r[0]
    return total_duration, duration
