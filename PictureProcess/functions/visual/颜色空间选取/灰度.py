import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import skimage.morphology as sm
image = cv2.imread('../1.jpg', cv2.IMREAD_COLOR)
img1 = np.array(image, dtype='int')  # 转换成int型，不然会导致数据溢出
# 超绿灰度图
b, g, r = cv2.split(img1)
ExG = 2 * g - r - b
GR = g - r
[m, n] = ExG.shape

for i in range(m):
    for j in range(n):
        if ExG[i, j] < 0:
            ExG[i, j] = 0
        elif ExG[i, j] > 255:
            ExG[i, j] = 255
for i in range(m):
    for j in range(n):
        if GR[i, j] < 0:
            GR[i, j] = 0
        elif GR[i, j] > 255:
            GR[i, j] = 255

ExG = np.array(ExG, dtype='uint8')  # 重新转换成uint8类型
GR = np.array(GR,dtype='uint8')
ret2, th2 = cv2.threshold(GR, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

th2 = sm.closing(th2,sm.disk(9))
th2 = sm.opening(th2,sm.disk(9))
print(th2.shape)

w, h = th2.shape[:2]
mask = np.zeros([w + 2, h + 2], np.uint8)
for y in range(0, 305):
    for x in range(0, 250):
        if th2[y,x] == 0:
            cv2.floodFill(th2, mask, (x, y), 255, (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
for y in range(0,305):
    for x in range(422, 549):
        if th2[y,x] == 0:
            cv2.floodFill(th2, mask, (x, y), 255, (200, 200, 200), (100, 100, 100), cv2.FLOODFILL_FIXED_RANGE)
h, w = th2.shape[:2]
l1 = []
l2 = []
count = 0
for y in range(0, h):
    left = 0
    right = w - 1
    while left < w and th2[y,left] == 255:
        left = left + 1
    while right > 0 and th2[y, right] == 255:
        right = right - 1
    if left != w and right != 0:
        count += 1
        l1.append((left + right) // 2)
        l2.append(h - y)
        th2[y, (left + right) // 2] = 255
print(count)
l1.reverse()
l2.reverse()
# pengzhang = sm.dilation(th2,sm.square(5))
# fushi = sm.erosion(th2,sm.square(5))
# kaiyunsuan = sm.opening(th2,sm.disk(5))
# biyunsuan = sm.closing(th2,sm.disk(5))
# th3 = sm.closing(th2,sm.disk(5))
# xianbihoukai = sm.opening(th3,sm.disk(5))
#
# plt.figure(figsize=(10, 5), dpi=80)
# plt.subplot(141), plt.imshow(cv2.cvtColor(pengzhang, cv2.COLOR_BGR2RGB)), \
# plt.title('dilation'), plt.axis('off')
# plt.subplot(142), plt.imshow(cv2.cvtColor(fushi, cv2.COLOR_BGR2RGB)), \
# plt.title('erosion'), plt.axis('off')
# plt.subplot(143), plt.imshow(cv2.cvtColor(th2, cv2.COLOR_BGR2RGB)), \
# plt.title('OTSU'), plt.axis('off')
# plt.subplot(144), plt.imshow(cv2.cvtColor(xianbihoukai, cv2.COLOR_BGR2RGB)), \
# plt.title('closing & opening'), plt.axis('off')
#
#
# plt.show()
