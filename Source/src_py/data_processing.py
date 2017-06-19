from scipy.misc import imresize, imread, imshow, imsave
import os
from glob import glob
from tqdm import tqdm
import random
import numpy as np

np.random.seed(233)

with open("gtnames.txt", "r") as f:
    test_files = f.readlines()

test_files = [y.strip() for y in test_files]
print(test_files)
print(len(test_files))

all_img = glob("./input_resized/*.jpg")

X_train = list()
X_test = list()

for img_path in tqdm(all_img):
    img = imread(img_path)
    img_flat = img.reshape(96*96*3)
    img_name = os.path.basename(img_path)
    img_name = img_name.split('.')[0]
    if img_name in test_files:
        X_test.append(img_flat)
    else:
        X_train.append(img_flat)

all_img = glob("./input_gt_resized/*.bmp")

y_train = list()
y_test = list()

for img_path in tqdm(all_img):
    img = imread(img_path, True)
    img_flat = img.reshape(96*96)
    img_flat = np.where(img_flat > 0, 1.0, 0.0)
    img_name = os.path.basename(img_path)
    img_name = img_name.split('.')[0]
    if img_name in test_files:
        y_test.append(img_flat)
    else:
        y_train.append(img_flat)

X_test = np.array(X_test)
X_train = np.array(X_train)
y_test = np.array(y_test)
y_train = np.array(y_train)

print y_test.shape, X_test.shape
assert X_test.shape[0] == y_test.shape[0]
assert y_test.shape[0] == 150
np.save('X_test', X_test, allow_pickle=False)
np.save('y_test', X_test, allow_pickle=False)

# Take randomly 150 images as validation test
random_idx = random.sample(range(len(X_train)), 150)
left_idx = list(set(range(len(X_train))) - set(random_idx))

X_validation = X_train[random_idx]
y_validation = y_train[random_idx]

X_train = X_train[left_idx]
y_train = y_train[left_idx]

assert X_train.shape[0] == y_train.shape[0]
assert y_train.shape[0] == 700
np.save('X_train', X_train, allow_pickle=False)
np.save('y_train', y_train, allow_pickle=False)

assert X_validation.shape[0] == y_validation.shape[0]
assert y_validation.shape[0] == 150
np.save('X_validation', X_validation, allow_pickle=False)
np.save('y_validation', y_validation, allow_pickle=False)
