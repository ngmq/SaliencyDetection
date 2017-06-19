from scipy.misc import imresize, imread, imshow, imsave
import os
from glob import glob
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(233)

with open("gtnames.txt", "r") as f:
    test_files = f.readlines()

test_files = [y.strip() for y in test_files]
print(test_files)
print(len(test_files))

all_img = glob("./input_resized/*.jpg")
all_img = sorted(all_img)

X_train = list()
X_test = list()

cnt = 0

print("=====INPUT:=====")
for img_path in tqdm(all_img):
    img = imread(img_path)
    img_flat = img.reshape(96*96*3)
    img_name = os.path.basename(img_path)
    img_name = img_name.split('.')[0]
    if img_name in test_files:
        X_test.append(img_flat)
    else:
        X_train.append(img_flat)
        
    # print(img_name)
    # cnt += 1
    # if(cnt == 100):
        # break

all_img = glob("./input_gt_resized/*.bmp")
all_img = sorted(all_img)

y_train = list()
y_test = list()

cnt = 0
for img_path in tqdm(all_img):
    img = imread(img_path, True)
    img_flat = img.reshape(48*48)
    img_flat = np.where(img_flat > 0, 1.0, 0.0)
    img_name = os.path.basename(img_path)
    img_name = img_name.split('.')[0]
    if img_name in test_files:
        y_test.append(img_flat)
    else:
        y_train.append(img_flat)
        
    # print(img_name)
    # cnt += 1
    # if(cnt == 100):
        # break

X_test = np.array(X_test)
X_train = np.array(X_train)
y_test = np.array(y_test)
y_train = np.array(y_train)

y_values = set(y_test.flatten())
assert len(y_values) == 2

y_values = set(y_train.flatten())
assert len(y_values) == 2


print y_test.shape, X_test.shape
assert X_test.shape[0] == y_test.shape[0]
assert y_test.shape[0] == 150
np.save('X_test', X_test, allow_pickle=False)
np.save('y_test', y_test, allow_pickle=False)

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

# x_train = np.load('X_train.npy')
# nx = len(x_train)
# x_train = x_train.reshape(nx, 96, 96, 3)
# y_train = np.load('y_train.npy')
# y_train = y_train.reshape(nx, 48, 48)

# random_idx = random.sample(range(nx), 10)

# for i in range(10):
    # plt.imshow(X_train[i].reshape(96, 96, 3))
    # plt.show()
    # plt.imshow(y_train[i].reshape(48, 48))
    # plt.show()



