from scipy.misc import imresize, imread, imshow, imsave
import os
from glob import glob
from tqdm import tqdm

with open("gtnames.txt", "r") as f:
	x = f.readlines()
	

x = [y.strip() for y in x]
print(x)


all_img = glob("./input/*.jpg")
count = 0
for img_path in tqdm(all_img):
	img = imread(img_path)
	img = imresize(img, (96, 96, 3))
	img_name = os.path.basename(img_path)
	imsave("./input_resized/{0}".format(img_name), img)
