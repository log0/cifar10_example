import re
import csv
import glob
import numpy as np
import scipy
import scipy.ndimage

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn.cross_validation import *
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder

TRAIN_DIR = 'data/train/*'
TRAIN_LABELS = 'data/trainLabels.csv'
R_DIGITS = re.compile('\d+')
N_COUNT = 10000

reader = csv.reader(open(TRAIN_LABELS))
reader.__next__()

train_labels_map = {i[0]: i[1] for i in reader}
train_labels = []
train_data = []
train_files = glob.glob(TRAIN_DIR)
for i, train_file in enumerate(train_files):
    if i >= N_COUNT: break

    image = scipy.ndimage.imread(train_file)
    train_data.append(image)

    file_id = R_DIGITS.findall(train_file)[0]
    label = train_labels_map[file_id]
    train_labels.append(label)
train_data = np.array(train_data)
train_data = np.transpose(train_data, axes = (0, 3, 1, 2))
train_data = train_data / 255

train_labels = np.array(train_labels)
train_labels_encoder = LabelEncoder()
train_labels = train_labels_encoder.fit_transform(train_labels)
n_classes = len(set(train_labels))

raw_train_labels = train_labels
train_labels = np_utils.to_categorical(raw_train_labels)

X_train, X_cv, Y_train, Y_cv, raw_train_labels_train, raw_train_labels_cv = \
        train_test_split(train_data, train_labels, raw_train_labels, test_size = 0.20)

print(train_data.shape)
print(train_labels.shape)
print(raw_train_labels_train)

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode = 'same', input_shape = (3, 32, 32)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, border_mode = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode = 'same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-7, momentum=0.2, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train, nb_epoch = 20, batch_size = 50, show_accuracy = True, verbose = 1, validation_split = 0.05)
Y_cv_pred = model.predict_classes(X_cv, batch_size = 50, verbose = 1)

print('Y_cv: ', raw_train_labels_cv[:100])
print('Y_cv_pred: ', Y_cv_pred[:100])

print('Accuracy score: ', accuracy_score(Y_cv_pred, raw_train_labels_cv))
