from scipy.misc import imresize, imread, imshow, imsave
import os
from glob import glob
from tqdm import tqdm

with open("gtnames.txt", "r") as f:
	x = f.readlines()
	

x = [y.strip() for y in x]
print(x)


all_img = glob("./input_gt/*.bmp")
count = 0
for img_path in tqdm(all_img):
	img = imread(img_path, True)
	img = imresize(img, (48, 48))
	img_name = os.path.basename(img_path)
	imsave("./input_gt_resized/{0}".format(img_name), img)
