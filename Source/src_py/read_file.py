import cv2
import numpy as np
import yaml

tt = cv2.FileStorage('myfile.yml', cv2.FILE_STORAGE_READ)
skip_lines = 2
with open('myfile.yml') as f:
    for i in range(skip_lines):
        _ = f.readline()
    data = yaml.load(f)

#print(data)
#print(data['rows'])

m = np.asarray(data['data'])
print(m)
