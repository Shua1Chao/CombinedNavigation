import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter
from numpy.random import randn
from math import sqrt, pow, pi, sin, cos, tan


def PictureProcess(capture):
    # 获取一帧
    ret, img = capture.read()
    # 将这帧转换为灰度图
    r, g, b = cv2.split(img)
    ExG = 2 * g - r - b
    retval, T = cv2.threshold(ExG, 0, 255, cv2.THRESH_OTSU)  # 大津法自动阈值分割

    cv2.imwrite("2.jpg", T)
    im = Image.open('2.jpg')
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    data = im.getdata()
    w, h = im.size
    black_point = 0
    # 自制滤波函数
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
                if black_point >= 5:
                    im.putpixel((x, y), 255)
                black_point = 0
    # floodfill填充较小连接区域
    img = cv2.cvtColor(np.asanyarray(im), cv2.COLOR_RGB2BGR)  # 转换为imread格式
    w, h = img.shape[:2]
    mask = np.zeros([w + 2, h + 2], np.uint8)
    for x in range(0, 580):
        for y in range(0, 560):
            if data[w * x + y] == 255:
                cv2.floodFill(img, mask, (x, y), (0, 0, 0), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    for x in range(720, 1279):
        for y in range(0, 719):
            if data[w * x + y] == 255:
                cv2.floodFill(img, mask, (x, y), (0, 0, 0), (200, 200, 200), (100, 100, 100), cv2.FLOODFILL_FIXED_RANGE)

    # 去除较小联通区域
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为Image.open格式
    contours, hierarch = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 200:
            cv2.drawContours(img, contours, i, (0, 0, 0), -1)
    h, w = img.shape[:2]

    # 最小二乘法拟合
    l = []
    count = 0
    for y in range(0, h):
        left = 0
        right = w - 1
        while left < w and img.item(y, left, 0) == 0:
            left = left + 1
        while right > 0 and img.item(y, right, 0) == 0:
            right = right - 1
        if left != w and right != 0:
            count += 1
            z = [(left + right) // 2, y]
            l.append(z)
    return l


def twoD_threeD(u, v):
    camera_matrix = np.array([[1.64332228e+03,0,5.44861838e+02]
                              [0,2.08744299e+03,2.35903083e+02]
                              [0,0,1]])
    rvec = np.array([[0.707, -0.707, 0],
                     [0.707, 0.707, 0],
                     [0, 0, 1]])

    tvec = a = np.array([0, -1000, 0])

    # (R T, 0 1)矩阵
    Trans = np.hstack((rvec, [[tvec[0]], [tvec[1]], [tvec[2]]]))

    # 相机内参和相机外参 矩阵相乘
    temp = np.dot(camera_matrix, Trans)

    Pp = np.linalg.pinv(temp)

    # 点（u, v, 1) 对应代码里的 [605,341,1]
    p1 = np.array([u, v, 1], np.float)

    print("像素坐标系的点:", p1)

    X = np.dot(Pp, p1)

    print("X:", X)

    # 与Zc相除 得到世界坐标系的某一个点
    X1 = np.array(X[:3], np.float) / X[3]

    print("X1:", X1)

    return X1


def fx(x, dt):
    F = np.array([[1, 0, dt, 0, 0, 0, 0],
                  [0, 1, 0, dt, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1]], dtype=float)
    return np.dot(F, x)


def hx(x):
    u = x[2]
    v = x[3]
    phi = x[4]

    H = np.array([[1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, u / sqrt(u * u + v * v), v / sqrt(u * u + v * v), 0, 0, 0],
                  [-cos(phi), sin(phi), 0, 0, 0, cos(phi), -sin(phi)],
                  [-sin(phi), -cos(phi), 0, 0, 0, sin(phi), cos(phi)]], dtype=float)
    return np.dot(H, x)


def JW2GS(J, W):
    a = 6378137
    b = 6356752.3142
    f = (a - b) / a
    e = sqrt(2 * f - pow(f, 2))
    e1 = e / sqrt(1 - pow(e, 2))
    L0 = 120  # 中央经线
    W0 = 0  # 原点纬线
    k0 = 1
    FE = 500000
    FN = 0
    BR = (W - W0) * pi / 180  # 纬度弧长
    lo = (J - L0) * pi / 180  # 经差弧度
    N = a / sqrt(1 - pow((e * sin(BR)), 2))  # 卯酉圈曲率半径

    C = pow(a, 2) / b
    B0 = 1 - 3 * pow(e1, 2) / 4 + 45 * pow(e1, 4) / 64 - 175 * pow(e1, 6) / 256 + 11025 * pow(e1, 8) / 16384
    B2 = B0 - 1
    B4 = 15 / 32 * pow(e1, 2) - 175 / 384 * pow(e1, 6) + 3675 / 8192 * pow(e1, 8)
    B6 = 0 - 35 / 96 * pow(e1, 6) + 735 / 2048 * pow(e1, 8)
    B8 = 315 / 1024 * pow(e1, 8)
    s = C * (B0 * BR + sin(BR) * (B2 * cos(BR) + B4 * pow(cos(BR), 3) + B6 * pow(cos(BR), 5) + B8 * pow(cos(BR), 7)))
    t = tan(BR)
    g = e1 * cos(BR)
    XR = s + pow(lo, 2) / 2 * N * sin(BR) * cos(BR) + pow(lo, 4) * N * sin(BR) * pow((cos(BR)), 3) / 24 * (
            5 - pow(t, 2) + 9 * pow(g, 2) + 4 * pow(g, 4)) + pow(lo, 6) * N * sin(BR) * pow((cos(BR)), 5) * (
                 61 - 58 * pow(t, 2) + pow(t, 4)) / 720
    YR = lo * N * cos(BR) + pow(lo, 3) * N / 6 * pow((cos(BR)), 3) * (1 - pow(t, 2) + pow(g, 2)) + pow(lo,
                                                                                                       5) * N / 120 * pow(
        (cos(BR)), 5) * (5 - 18 * pow(t, 2) + pow(t, 4) + 14 * pow(g, 2) - 58 * pow(g, 2) * pow(t, 2))
    X = YR + FE
    Y = XR + FN
    return X, Y


def getJW():
    # TODO
    J = 1
    W = 2
    return J, W


def main():
    dt = 1.0
    points = MerweScaledSigmaPoints(7, alpha=1, beta=2, kappa=-1)

    kf = UnscentedKalmanFilter(dim_x=7, dim_z=6, dt=dt, fx=fx, hx=hx, points=points)
    kf.x = np.array([0, 0, 0.3, 0, 0, 1, 1])
    kf.P *= 1
    # z_std = 0.1
    kf.R = np.array([[0.0025, 0, 0, 0, 0, 0],
                     [0, 0.0025, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0.01, 0, 0],
                     [0, 0, 0, 0, 0.09, 0],
                     [0, 0, 0, 0, 0, 0.09]])
    kf.Q *= 0.01

    l = PictureProcess(0)
    zsl = [[0,0,1.570796327,0.3,328,1],[0.3,0.3,0.927295218,0.3,331,9]]
    for z in zsl:
        X1 = twoD_threeD(z[0], z[1])
        Yv, Xv = getJW()
        # 利用 GPS 获取行驶角度值 "alpha"
        # 利用 Xp = Xv + X1[0] * cos (alpha) + X1[1] * sin (alpha)，Yp = Yv - X1[0] * sin (alpha) + X1[1] * cos (alpha) 来计算"视觉目标点"在"大地坐标系"中的坐标
        #zs = [Xv, Yv, alpha, 0.3, Xp, Yp]
        kf.predict()
        kf.update(z)
        print(kf.x, 'log-likelihood', kf.log_likelihood)
if __name__ == '__main__':
    main()

r'''
    zs = [[1 + randn() * z_std, i + randn() * z_std, i + randn() * z_std, i + randn() * z_std, i + randn() * z_std,
           i + randn() * z_std] for i in range(50)]

    for z in zs:
        kf.predict()
        kf.update(z)
        print(kf.x, 'log-likelihood', kf.log_likelihood)'''

r'''
def fx(x, dt):
    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]], dtype=float)

    return np.dot(F, x)


def hx(x):
    return np.array([x[0], x[2]])


dt = 0.1
points = MerweScaledSigmaPoints(4,alpha = 1,beta=2,kappa=-1)
kf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt,
                           fx=fx, hx=hx, points=points)
kf.P *= 0.2
z_std = 0.1
kf.R = np.diag([z_std**2,z_std ** 2])
kf.Q = Q_discrete_white_noise(dim=2,dt=dt,var=0.01**2,block_size=2)

zs = [[1 + randn()*z_std, i+randn()*z_std] for i in range(50)]

for z in zs:
    kf.predict()
    kf.update(z)
    print(kf.x,'log-likelihood',kf.log_likelihood) '''
