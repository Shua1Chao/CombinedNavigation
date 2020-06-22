import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

img = cv2.imread("7.jpg")
h,w = img.shape[:2]
print(w,h)
l1 = []
l2 = []
for y in range(0,h):
    left = 0
    right = w - 1
    while left < w and img.item(y,left,0) == 0:
        left = left + 1
    while right > 0 and img.item(y,right,0) == 0:
        right = right - 1
    print(left,right)
    if left != w and right != 0 :
        l1.append((left + right) // 2)
        l2.append(h - y)
        img.itemset((y, (left + right) // 2, 0), 0)
        img.itemset((y, (left + right) // 2, 1), 0)
        img.itemset((y, (left + right) // 2, 2), 0)
x = np.array(l1)
y = np.array(l2)
def Least_squares(x,y):
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    for i in np.arange(50):
        k = (x[i]-x_)* (y[i]-y_)
        m += k
        p = np.square( x[i]-x_ )
        n = n + p
    a = m/n
    b = y_ -  a * x_
    return a,b
if __name__ == '__main__':
    a,b = Least_squares(x,y)
    print(a,b)
    y1 = a * x + b
    plt.figure(figsize=(10, 5), facecolor='w')
    plt.plot(x, y, 'ro', lw=2, markersize=6)
    plt.plot(x, y1, 'r-', lw=2, markersize=6)
    plt.grid(b=True, ls=':')
    plt.xlabel(u'X', fontsize=16)
    plt.ylabel(u'Y', fontsize=16)
    plt.show()
    cv2.imshow("1",img)
    cv2.waitKey(0)