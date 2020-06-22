import numpy as np
from math import sin, cos
import cv2


# TODO
def twoD_threeD(u, v):
    camera_matrix = np.array([[fx, 0, Cx],
                              [0, fy, Cy],
                              [0, 0, 1]])
    rvec = np.array([[1, 0, 0],
                     [0，cos(Phi), -sin(Phi)],
                     [0，sin(Phi), cos(Phi)]])

    tvec = a = np.array([0, -H, 0])

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

def twoD_threeD1(u, v):
    camera_matrix = np.array([[1, 0, 1],
                              [0, 1, 1],
                              [0, 0, 1]])
    rvec = np.array([[1, 0, 0],
                     [0, 0, -1],
                     [0, 1, 0]])

    tvec = a = np.array([0, -0.3, 0])

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
u  = 1
v = 2
print(twoD_threeD1(u,v))