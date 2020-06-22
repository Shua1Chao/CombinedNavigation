import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
########     四个不同的滤波器    #########
img = cv2.imread('1.jpg')

imgInfo = img.shape
height = imgInfo[0] - 1  # 防止越界
width = imgInfo[1] - 1

temp = 2000  # 噪声点的个数
for i in range(0, temp):
    if random.randint(1, temp) % 2 == 0:
        img[random.randint(0, height), random.randint(0, width)] = (255, 255, 255)
    if random.randint(1, temp) % 2 != 0:
        img[random.randint(0, height), random.randint(0, width)] = (0, 0, 0)
cv2.imshow('dst', img)
cv2.imwrite('noise.jpg', img)

# 均值滤波
img_mean = cv2.blur(img, (3,3))

# 高斯滤波
img_Guassian = cv2.GaussianBlur(img,(3,3),0)

# 中值滤波
img_median = cv2.medianBlur(img, 3)

# 双边滤波
img_bilater = cv2.bilateralFilter(img,9,75,75)

# 展示不同的图片
titles = ['mean', 'Gaussian', 'median', 'bilateral']
imgs = [ img_mean, img_Guassian, img_median, img_bilater]
cv2.imshow('mean', img_mean)
cv2.imshow('Gaussian', img_Guassian)
cv2.imshow('median', img_median)
cv2.imshow('bilateral', img_bilater)
cv2.waitKey(0)
