import matplotlib.pyplot as plt
import random
import numpy as np
import keras
from keras.models import load_model, Sequential, Model
from scipy.misc import imshow, imsave, imread, imresize
from tqdm import tqdm
from glob import glob
import os

np.random.seed(233)

with open("gtnames.txt", "r") as f:
    test_files = f.readlines()

test_files = [y.strip() for y in test_files]
print(test_files)
print(len(test_files))

input_size = (96, 96, 3)
output_size = (48, 48)
nb_output = output_size[0] * output_size[1]

print(nb_output)

model = load_model('model.h5')

# x_train = np.load('X_train.npy')
# nx = len(x_train)
# x_train = x_train.reshape(nx, 96, 96, 3)

# x_validation = np.load('X_validation.npy')
# nx = len(x_validation)
# x_validation = x_validation.reshape(nx, 96, 96, 3)

# x_test = np.load('X_test.npy')
# nx = len(x_test)
# x_test = x_test.reshape(nx, 96, 96, 3)

# y_train = np.load('y_train.npy')
# y_validation = np.load('y_validation.npy')
# y_test = np.load('y_test.npy')

# y_values = set(y_test.flatten())
# assert len(y_values) == 2

# result = model.evaluate(x_test, y_test)
# print("metrics: {0}".format(model.metric_names))
# print("result = {0}".format(result))

# y = model.predict(x_test)
# print(len(y))

all_img = glob("./input/*.jpg")

print("Ok here")

for img_path in tqdm(all_img):
    img = imread(img_path)
    original_shape = img.shape
    img = imresize(img, (96, 96, 3))
    
    img_out = model.predict(np.array([img]))[0]
    img_out = np.rint(img_out)
    img_out = img_out.reshape(48, 48)
    img_out = imresize(img_out, (original_shape[0], original_shape[1]))
    
    img_name = os.path.basename(img_path)
    img_name = img_name.split('.')[0]
    
    if img_name in test_files:
        imsave("./method_DL/{0}.png".format(img_name), img_out)





