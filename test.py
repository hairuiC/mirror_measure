import cv2
import os
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
from functools import partial
import numpy as np
path = '/home/jing/Documents/files'
filelist = []
for filenames in os.listdir(path):
    filelist.append(filenames)

img = cv2.imread('/home/jing/Documents/fuck.jpg')
height, width = img.shape[:2]
img = cv2.resize(img, (int(width/2), int(height/2)))
# img[0, 0, :] = 255
cv2.imshow('what', img)
cv2.waitKey(0)

pts = np.array([[1666, 137],
              [3648, 187],
              [3600, 3360],
              [1600, 3310]])
pts = np.array([pts])
mask = np.zeros(img.shape[:2], np.uint8)
print(mask.shape)

cv2.polylines(mask, pts, 1, 255)
cv2.fillPoly(mask, pts, 255)
dst = cv2.bitwise_and(img, img, mask=mask)


bg = np.ones_like(img, np.uint8) * 255
cv2.bitwise_not(bg, bg, mask=mask)  # bg的多边形区域为0，背景区域为255
dst_white = bg + dst

cv2.imwrite("dst.jpg", dst)
cv2.imwrite("dst1.jpg", dst_white)

