import re
import csv
import glob
import numpy as np
import pandas as pd
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

from PIL import Image, ImageDraw

def to_int_array(a):
    return list(map(lambda i: int(i), a))

TRAIN_DIR = 'data/train/*'
TRAIN_LABELS = 'data/labels.csv'
R_DIGITS = re.compile('\d+')
N_COUNT = 10000

reader = csv.reader(open(TRAIN_LABELS))

train_labels_map = {i[0]: to_int_array(i[1:]) for i in reader}
train_labels = []
train_data = []
train_files = glob.glob(TRAIN_DIR)
ids = []
for i, train_file in enumerate(train_files):
    if i >= N_COUNT: break

    image = scipy.ndimage.imread(train_file)
    train_data.append(image)

    file_id = R_DIGITS.findall(train_file)[0]
    label = train_labels_map[file_id]
    train_labels.append(label)
    ids.append(file_id)
ids = np.array(ids).astype(int)
train_data = np.array(train_data)
train_data = np.transpose(train_data, axes = (0, 3, 1, 2))
train_data = train_data / 255
train_labels = np.array(train_labels)
train_labels = train_labels / 32

X_train, X_cv, Y_train, Y_cv, ids_train, ids_cv = train_test_split(train_data, train_labels, ids, test_size = 0.20)

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode = 'same', input_shape = (3, 32, 32)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, border_mode = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(4))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-7, momentum=0.2, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train, Y_train, nb_epoch = 20, batch_size = 50, show_accuracy = True, verbose = 1, validation_split = 0.05)
Y_cv_pred = model.predict(X_cv, batch_size = 50, verbose = 1)

print('Y_cv.shape: ', Y_cv.shape)
print('Y_cv_pred.shape: ', Y_cv_pred.shape)
print('ids_cv.shape:' , ids_cv.shape)

some_Y_cv = pd.DataFrame(np.round(Y_cv[:20] * 32)).set_index(ids_cv[:20])
some_Y_cv_pred = pd.DataFrame(np.round(Y_cv_pred[:20] * 32)).set_index(ids_cv[:20])
some = pd.concat([some_Y_cv, some_Y_cv_pred], axis = 1)
# print('Y_cv: \n', some_Y_cv)
# print('Y_cv_pred: \n', some_Y_cv_pred)
print(some)

Y_cv *= 32
Y_cv_pred *= 32

for i, id in enumerate(ids_cv):
    filepath = 'data/train/%s.png' % (id)
    image = Image.open(filepath)
    draw = ImageDraw.Draw(image)
    xys = ((Y_cv_pred[i][0], Y_cv_pred[i][1]), (Y_cv_pred[i][2], Y_cv_pred[i][3]))
    draw.ellipse(xys, fill = 'blue')

    save_filepath = 'data/predictions/%s.png' % (id)
    image.save(save_filepath)
