import os
import re
import csv
import glob
import numpy as np
import random
import scipy
import scipy.ndimage

from PIL import Image, ImageDraw

def random_bounding_circle():
    IMAGE_WIDTH = 32
    IMAGE_HEIGHT = 32

    MAX_RADIUS = 5

    x0 = random.randint(0, IMAGE_WIDTH - MAX_RADIUS)
    y0 = random.randint(0, IMAGE_HEIGHT - MAX_RADIUS)
    # radius = random.randint(1, MAX_RADIUS)
    radius = random.randint(MAX_RADIUS, MAX_RADIUS)
    x1 = x0 + radius
    y1 = y0 + radius

    return ((x0, y0), (x1, y1))

R_DIGITS = re.compile('\d+')
INPUT_DIR = 'data/source/*'
OUTPUT_DIR = 'data/train/'
LABELS_PATH = 'data/labels.csv'

N_COUNT = 10000

source_data = []
source_files = glob.glob(INPUT_DIR)

train_data = {}
labels = {}
for i, source_file in enumerate(source_files):
    if i >= N_COUNT: break
    file_id = R_DIGITS.findall(source_file)[0]

    image = Image.open(source_file)
    draw = ImageDraw.Draw(image)

    x0y0x1y1 = random_bounding_circle()
    draw.ellipse(x0y0x1y1, fill = 'red')

    (x0, y0), (x1, y1) = x0y0x1y1

    train_data[file_id] = image
    labels[file_id] = [x0, y0, x1, y1]

for file_id, image in train_data.items():
    filename = os.path.join(OUTPUT_DIR, str(file_id) + '.png')
    image.save(filename)

# Save labels data
with open(LABELS_PATH, 'w') as f_labels:
    writer = csv.writer(f_labels)
    for file_id, label in labels.items():
        row = [file_id]
        row.extend(label)
        writer.writerow(row)
