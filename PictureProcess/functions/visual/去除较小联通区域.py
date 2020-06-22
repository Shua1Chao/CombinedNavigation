import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from PIL import Image, ImageEnhance, ImageFilter
from skimage import morphology,measure,io

img_name = "2.jpg"
img = cv2.imread(img_name)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
contours,hierarch=cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area < 2:
        cv2.drawContours(img,contours,i,(0,0,0),-1)
cv2.imshow("1",img)
cv2.waitKey(0)