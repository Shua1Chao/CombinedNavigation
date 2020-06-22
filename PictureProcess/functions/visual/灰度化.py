import cv2
import time
from PIL import Image, ImageEnhance
import numpy as np

clock1 = time.clock()

img = cv2.imread("1.jpg")
r,g,b = cv2.split(img)
ExG = 2 * g - r - b
retval, T = cv2.threshold(ExG, 0, 255, cv2.THRESH_OTSU)
cv2.imwrite("2.jpg",T)

im = Image.open('2.jpg')
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
            if top_pixel_right == 255:
                black_point += 1
            if down_pixel_left == 255:
                black_point += 1
            if down_pixel_right == 255:
                black_point += 1
            if black_point >= 4:
                im.putpixel((x, y), 255)
            black_point = 0
img = cv2.cvtColor(np.asanyarray(im), cv2.COLOR_RGB2BGR)  # 转换为imread格式
w, h = img.shape[:2]
mask = np.zeros([w + 2, h + 2], np.uint8)
for x in range(0, 280):
    for y in range(0, 240):
        if data[w * x + y] == 255:
            cv2.floodFill(img, mask, (x, y), (0, 0, 0), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
for x in range(360, 639):
    for y in range(0, 240):
        if data[w * x + y] == 255:
            cv2.floodFill(img, mask, (x, y), (0, 0, 0), (200, 200, 200), (100, 100, 100), cv2.FLOODFILL_FIXED_RANGE)
cv2.imshow("1",img)
# 去除较小联通区域
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为Image.open格式
contours, hierarch = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area < 20:
        cv2.drawContours(img, contours, i, (0, 0, 0), -1)
cv2.imshow("2",img)
h, w = img.shape[:2]
# 最小二乘法拟合
l1 = []
l2 = []
print(w, h)
count = 0
for y in range(0, h):
    left = 0
    right = w - 1
    while left < w and img.item(y, left, 0) == 0:
        left = left + 1
    while right > 0 and img.item(y, right, 0) == 0:
        right = right - 1
    if left != w and right != 0:
        print(left, right)
        count += 1
        l1.append((left + right) // 2)
        l2.append(y)
x = np.array(l1)
y = np.array(l2)
x_ = x.mean()
y_ = y.mean()
m = np.zeros(1)
n = np.zeros(1)
k = np.zeros(1)
p = np.zeros(1)
for i in np.arange(count):
    k = (x[i] - x_) * (y[i] - y_)
    m += k
    p = np.square(x[i] - x_)
    n = n + p
a = m / n
b = y_ - a * x_
y1 = a * x + b