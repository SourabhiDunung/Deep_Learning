import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'microsoft-catsvsdogs-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F550917%2F1003830%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240312%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240312T081930Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D01764e19b2c53ea9dc838b5f4860af69460ef45c16d6a8abb59553eef11e3e7011e8114453e8a1b01be529a890cbb8b7de9b1010769174fe8f64b1d3487c79e00b77ecda91d09c5ad646edbd93598c1e8c42ce6d11e34e79ea4b57ed4a5b8ef4bb046ca381d37b39fe4bf3922a4894584c6ab8a594db9afebc63da67523c8a2bd0ecae9338ccebc6108ba65aabfe8c3d34840d77132605bc8d3b3d115d449a4a63a295333948ef1d9ada50c62b27a5810106cf602be82b04dab8160a6900098672dc9286790f2769295051320cab083ab6fe1e4ab30db601534dc9f58f0afd4b47eaec86f14c2d50c612615fe119890a92609b4ac8b313038d44636e57559128'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import PIL
import pathlib

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout, Flatten,Activation, BatchNormalization,MaxPooling2D
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

"""#2.Load dataset"""

data_dir = pathlib.Path("../input/microsoft-catsvsdogs-dataset/PetImages")
print(data_dir)
list(data_dir.glob('*/*.jpg'))[:5]

dogs = list(data_dir.glob('Dog/*'))
dogs[:5]

PIL.Image.open(str(dogs[1]))

cats = list(data_dir.glob('Cat/*'))
cats[:5]

PIL.Image.open(str(cats[1]))

image_count=len(list(data_dir.glob('*/*.jpg')))
print(image_count)

"""#3."""

pet_images_dict={
    'cats':list(data_dir.glob('Cat/*')),
    'dogs':list(data_dir.glob('Dog/*')),
}
pet_labels_dict={
    'cats':0,
    'dogs':1,
}

IMAGE_WIDTH=128
IMAGE_HEIGHT=128
X,Y=[], []

for pet_name, images in pet_images_dict.items():
  print(pet_name)
  for image in images:
    img = cv2.imread(str(image))
    if isinstance(img,type(None)):
       #print('image not found')
       continue

    elif((img.shape[0]>=IMAGE_HEIGHT) and (img.shape[1]>=IMAGE_WIDTH)):
      resized_img = cv2.resize(img,(IMAGE_WIDTH, IMAGE_HEIGHT))
      X.append(resized_img)
      Y.append(pet_labels_dict[pet_name])

    else:
      #print('Invalid Image')
      continue

print(img.shape[1])

"""#4. Training the data"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

x=np.array(X)
y=np.array(Y)
trainX,testX,trainY,testY= train_test_split(x,y, train_size=0.8)

"""#5. Build CNN model and train it"""

model = tf.keras.models.Sequential()
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
IMAGE_CHANNELS=3

model.add(Conv2D(16,kernel_size=3, activation="relu", input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32,kernel_size=3, activation="relu", input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64,kernel_size=3, activation="relu", input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

model.fit(trainX,trainY,epochs=3)