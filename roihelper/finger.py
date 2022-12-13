import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt


def edgeDetection(gray, alpha=1.):
    """LoG边缘检测"""
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    absX = cv.convertScaleAbs(laplacian, alpha=alpha)
    absX = cv.GaussianBlur(absX, (3, 3), 100)
    """cv.imshow("", absX)
    cv.waitKey()"""
    h, w = absX.shape
    return absX, h, w


def getBoundary(absX):
    """取边缘检测后边界的点"""
    h, w = absX.shape
    ret, absX = cv.threshold(absX, 0, 255, cv.THRESH_OTSU)  # 二值化
    """plt.imshow(absX, cmap="gray")
    plt.show()"""
    t, t1 = 0, h - 1
    up_x, down_x, up_y, down_y = [], [], [], []  # 上下边界
    # 遍历图像, 逼近手指边缘点
    for i in range(w):
        y_min, y_max = 0, h - 1
        for j in range(h):
            if absX[j, i] > ret:
                if h // 2 > j > y_min:
                    y_min = j
                if h // 2 < j < y_max:
                    y_max = j
                    break
        t = max(t, y_min)
        t1 = min(t1, y_max)
        if y_max != h - 1:
            up_x.append(i)
            up_y.append(y_max)
        if y_min != 0:
            down_x.append(i)
            down_y.append(y_min)
    # t2 = (t + t1) / 2.
    up_x = np.array(up_x)
    down_x = np.array(down_x)
    up_y = np.array(up_y)
    down_y = np.array(down_y)
    return up_x, down_x, up_y, down_y


def fun2ploy(x, n):
    """
    数据转化为[x^0,x^1,x^2,...x^n], 首列变1
    :param x:
    :param n:
    :return:
    """
    lens = len(x)
    X = np.ones([1, lens])
    for i in range(1, n):
        X = np.vstack((X, np.power(x, i)))  # 按行堆叠
    return X


def getLine(absX):
    """
    拟合中心线
    :param absX:
    :return:
    """
    up_x, down_x, up_y, down_y = getBoundary(absX)
    line1 = [np.array([x, y]) for x, y in zip(up_x, up_y)]
    line2 = [np.array([x, y]) for x, y in zip(down_x, down_y)]
    line1 = np.asarray(line1)
    line2 = np.asarray(line2)
    res1 = cv.fitLine(line1, cv.DIST_L2, 0., 0.01, 0.01).reshape(-1)
    res2 = cv.fitLine(line2, cv.DIST_L2, 0., 0.01, 0.01).reshape(-1)
    k1 = res1[1] / res1[0]
    b1 = res1[3] - res1[2] * k1
    k2 = res2[1] / res2[0]
    b2 = res2[3] - res2[2] * k2
    k = (k1 + k2) / 2.
    b = (b1 + b2) / 2.
    return k, b


def getTheta(k, b, w):
    """计算旋转角度, 中心"""
    center_w = w / 2
    center_h = k * center_w + b
    """x = np.arange(w)
    y = k * x + b
    est_data = plt.plot(x,y,color="r",linewidth= 3)"""
    theta = math.atan(k)
    return theta, center_w, center_h


def rotateImg(img, theta, center_h, center_w):
    """旋转图像"""
    h, w, _ = img.shape
    rotate_mat = cv.getRotationMatrix2D((center_h, center_w), 180 * theta / math.pi, 1)
    ret = cv.warpAffine(img, rotate_mat, (w, h))
    return ret


def getRotateCenter(x, y, theta, center_w, center_h):
    """转换为旋转图像的坐标"""
    (cX, cY) = (center_w, center_h)
    if (cX - x) == 0:
        phi = math.atan((cY - y) / 1)
        sec = 1 / math.cos(phi)
    else:
        phi = math.atan((cY - y) / (cX - x))
        sec = (cX - x) / math.cos(phi)
    new_x = cX - math.cos(theta + phi) * sec
    new_y = cY - math.sin(theta + phi) * sec
    return round(new_x), round(new_y)


def verticalBound(theta, down_x, down_y, up_x, up_y, center_w, center_h):
    """获取旋转后的上下边界"""
    loc_down = list(zip(down_x, down_y))
    loc_up = list(zip(up_x, up_y))
    new_loc_down = list(map(lambda loc: getRotateCenter(loc[0], loc[1], theta,
                                                        center_w, center_h), loc_down))
    new_loc_up = list(map(lambda loc: getRotateCenter(loc[0], loc[1], theta,
                                                      center_w, center_h), loc_up))
    new_loc_down = list(zip(*new_loc_down))
    new_loc_up = list(zip(*new_loc_up))
    Y_max = int(max(new_loc_down[1]))
    Y_min = int(min(new_loc_up[1]))
    Y_max = np.array(Y_max)
    Y_min = np.array(Y_min)
    """Y1 = Y_max.repeat(len(up_x))
    Y2 = Y_min.repeat(len(down_x))
    plt.plot(down_x, Y2 + 2, color='g', linewidth=2)
    plt.plot(up_x, Y1 + 2, color='g', linewidth=2)"""
    return Y_max, Y_min


def horizontalBound(cropped):
    """左右边界"""
    max_1, max_2 = 0, 0
    height, width, _ = cropped.shape
    kernel_size = width // 20
    light = [0] * width
    temp = 0
    for w in range(width - kernel_size):
        light[w + kernel_size // 2] = cropped[:, w: w + kernel_size].sum()
        if w + kernel_size // 2 > width / 3:
            # 通过寻找峰值截取左边界
            if light[w + kernel_size // 2] > light[w + kernel_size // 2 + 1] and \
                    light[w + kernel_size // 2] > light[w + kernel_size // 2 - 1]:
                if max_1 < light[w + kernel_size // 2]:
                    max_1 = light[w + kernel_size // 2]
                    temp = w + kernel_size // 2
    # 对称性获取右边界
    cut_1, cut_2 = int(temp / 3), int(2 / 3 * (width - temp) + temp)
    """plt.figure()
    plt.plot(x, light, color="b", linewidth=2)
    plt.show()"""
    return cut_1, cut_2


def getROI(img, alpha=1.):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    # 边缘检测, 得到上下边界
    absX, h, w = edgeDetection(gray, alpha=alpha)
    up_x, down_x, up_y, down_y = getBoundary(absX)
    # 拟合上下边界的直线, 取平均得到中线
    k, b = getLine(absX)
    # 根据中线旋转恢复图像
    theta, center_w, center_h = getTheta(k, b, w)  # 旋转角度, 中心
    rotate_img = rotateImg(img, theta, center_h, center_w)

    # plt.imshow(img, cmap='gray')
    Y_max, Y_min = verticalBound(-theta, down_x, down_y, up_x, up_y, center_w, center_h)
    X_min, X_max = horizontalBound(rotate_img[Y_max:Y_min, :])
    roi = rotate_img[Y_max:Y_min, X_min:X_max]
    """cv.imshow("dst", roi)
    cv.imshow("src", img)
    cv.waitKey()"""
    return roi
