from model.utils import *
import cv2

img1 = cv2.imread('imgs/gain5shutter50ms/capture_003.jpg')
img2 = cv2.imread('imgs/gain40shutter50ms/capture_003.jpg')

grey_1 = calculate_lumitexel(img1)
grey_2 = calculate_lumitexel(img2)

xor = img_xor(grey_1, grey_2)
cv2.imshow('abc', xor)
cv2.waitKey(0)
