import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from PIL import Image, ImageEnhance, ImageFilter
from skimage import morphology

img_name = '2.jpg'
# 去除干扰线
im = Image.open(img_name)
clock1 = time.time()
# 图像二值化
enhancer = ImageEnhance.Contrast(im)
im = enhancer.enhance(2)
data = im.getdata()
w, h = im.size
black_point = 0
for x in range(1, w - 1):
    for y in range(1, h - 1):
        mid_pixel = data[w * y + x]  # 中央像素点像素值
        if mid_pixel == 0:  # 找出上下左右四个方向像素点像素值
            top_pixel = data[w * (y - 1) + x]
            top_pixel_left = data[w * (y - 1) + x - 1]
            top_pixel_right = data[w * (y - 1) + x + 1]
            left_pixel = data[w * y + (x - 1)]
            down_pixel = data[w * (y + 1) + x]
            down_pixel_left = data[w * (y + 1) + x - 1]
            down_pixel_right = data[w * (y + 1) + x + 1]
            right_pixel = data[w * y + (x + 1)]

            # 判断上下左右的黑色像素点总个数
            if top_pixel == 255:
                black_point += 1
            if left_pixel == 255:
                black_point += 1
            if down_pixel == 255:
                black_point += 1
            if right_pixel == 255:
                black_point += 1
            if top_pixel_left == 255:
                black_point += 1
            if top_pixel_left == 255:
                black_point += 1
            if down_pixel_left == 255:
                black_point += 1
            if down_pixel_right == 255:
                black_point += 1
            if black_point >= 5 :
                im.putpixel((x, y), 255)
            # print black_point
            black_point = 0

clock2 = time.time()
print(clock2-clock1)
img = cv2.cvtColor(np.asanyarray(im),cv2.COLOR_RGB2BGR)
w,h = img.shape[:2]
mask = np.zeros([w+2,h+2],np.uint8)
for x in range(0,254):
    for y in range(0,303):
        if data[w * x + y] == 255:
            cv2.floodFill(img, mask, (x, y), (0, 0, 0), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
for x in range(420,549):
    for y in range(0,303):
        if data[w * x + y] == 255:
            cv2.floodFill(img, mask, (x, y), (0, 0, 0), (200, 200, 200), (100, 100, 100), cv2.FLOODFILL_FIXED_RANGE)
cv2.imshow("2",img)
cv2.imwrite("6.jpg",img)
clock3 = time.time()
print(clock3-clock2)
print(clock3-clock1)
cv2.waitKey(0)