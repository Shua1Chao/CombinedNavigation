import numpy as np
from math import sqrt, pow, pi, sin, cos, tan


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
    BR = (W - W0) * pi / 180 # 纬度弧长
    lo = (J - L0) * pi / 180 # 经差弧度
    N = a / sqrt(1 - pow((e * sin(BR)), 2)) # 卯酉圈曲率半径

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
    X = YR + FE;
    Y = XR + FN;
    return X, Y


if __name__ == "__main__":
    print(JW2GS(118.23876000,39.71572000))
    print(JW2GS(118.23876012,39.71572300))

    print(JW2GS(118.23877150,39.71568717))
    print(JW2GS(118.23879217,39.71570483))
    print(JW2GS(118.23875800,39.71573500))
    print(JW2GS(118.23876000,39.71572033))
    print(JW2GS(118.23876113,39.71570200))
    print(JW2GS(118.23874700,39.71571350))
    print(JW2GS(118.23875567,39.71573083))
    print(JW2GS(118.23877017,39.71574567))
    print("-----------------------------------------")
    print(JW2GS(118.23877017, 39.71574841))
    print(JW2GS(118.23877017, 39.71575204))
    print(JW2GS(118.23877017, 39.71575610))
    print(JW2GS(118.23877017, 39.71575892))
    print(JW2GS(118.23877017, 39.71575989))
    print(JW2GS(118.23877017, 39.71575136))
    print(JW2GS(118.23877017, 39.71576642))
    print(JW2GS(118.23877017, 39.71577010))
    print(JW2GS(118.23877017, 39.71577361))
    print(JW2GS(118.23877017, 39.71575201))
    print(JW2GS(118.23877017, 39.71577963))
    print(JW2GS(118.23877017, 39.71578241))
    print(JW2GS(118.23877017, 39.71578632))
    print(JW2GS(118.23877017, 39.71578563))
    print(JW2GS(118.23877017, 39.71578914))
    print(JW2GS(118.23877017, 39.71578995))
    print(JW2GS(118.23877017, 39.71579268))
    print(JW2GS(118.23877017, 39.71579763))
    print(JW2GS(118.23877017, 39.71580042))
    print(JW2GS(118.23877017, 39.71580379))
    print(JW2GS(118.23877017, 39.71580812))
    print(JW2GS(118.23877017, 39.71581094))
    print(JW2GS(118.23877017, 39.71581456))
    print(JW2GS(118.23877017, 39.71581820))
    print(JW2GS(118.23877017, 39.71582456))
    print(JW2GS(118.23877017, 39.71581100))
    print(JW2GS(118.23877017, 39.71580563))
    print(JW2GS(118.23877017, 39.71580946))
    print(JW2GS(118.23877017, 39.71581310))
    print(JW2GS(118.23877017, 39.71581687))
    print(JW2GS(118.23877017, 39.71581999))
    print(JW2GS(118.23877017, 39.71582200))
    print(JW2GS(118.23877017, 39.71579512))
    print(JW2GS(118.23877017, 39.71573357))
    print(JW2GS(118.23877017, 39.71583460))
    print(JW2GS(118.23877017, 39.71583702))
    print(JW2GS(118.23877017, 39.71584174))
    print(JW2GS(118.23877017, 39.71584521))
    print(JW2GS(118.23877017, 39.71584815))
    print(JW2GS(118.23877017, 39.71584936))
    print(JW2GS(118.23877017, 39.71584720))
    print(JW2GS(118.23877017, 39.71585163))
    print(JW2GS(118.23877017, 39.71585478))
    print(JW2GS(118.23877017, 39.71585369))
    print(JW2GS(118.23877017, 39.71585620))
    print(JW2GS(118.23877017, 39.71585939))
    print(JW2GS(118.23877017, 39.71586204))
    print(JW2GS(118.23877017, 39.71586610))
    print(JW2GS(118.23877017, 39.71586899))
    print(JW2GS(118.23877017, 39.71578863))
    print(JW2GS(118.23877017, 39.71586525))
    print(JW2GS(118.23877017, 39.71582238))



