import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def show(img):
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])


def getMaxRegion(contours):
    areas = []
    for c in contours:
        areas.append(abs(cv.contourArea(c, False)))
    ret = contours[areas.index(max(areas))]
    return ret


def getROI(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    # 截取可能的手腕
    # temp = np.zeros((h + 160, w), np.uint8)
    # temp = [80:-80, :] = img_original
    temp = gray.copy()
    # 滤波 + 二值化
    # blur = cv.GaussianBlur(temp, (5, 5), 0)
    _, binary = cv.threshold(temp, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # binary = cv.GaussianBlur(binary, (5, 5), 0)
    # _, binary = cv.threshold(temp, 10, 255, cv.THRESH_BINARY)
    """plt.subplot(111)
    show(binary)
    plt.show()"""
    # 根据图像的矩计算中心
    M = cv.moments(binary)
    h, w = temp.shape
    x_c = M['m10'] // M['m00']
    y_c = M['m01'] // M['m00']
    """# 作出中心点
    # plt.figure(figsize=(15, 5))
    plt.subplot(111)
    show(binary)
    plt.plot(x_c, y_c, 'bx', markersize=10)
    plt.show()"""
    """  # 腐蚀操作可能导致轮廓不连贯, 不采用
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]]).astype(np.uint8)
    erosion = cv.erode(th, kernel, iterations=1)"""
    # 取出手部轮廓
    boundary = binary  # - erosion
    contours, _ = cv.findContours(boundary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    max_contour = getMaxRegion(contours)
    """# 作出轮廓
    img_color = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    img_cnt = cv.drawContours(img_color, [max_contour], -1, (255, 0, 0), 2)
    plt.plot(x_c, y_c, 'bx', markersize=10)
    show(img_cnt)
    plt.tight_layout()
    plt.show()"""
    max_contour = max_contour.reshape(-1, 2)
    left_id = max_contour.sum(-1).argmin()
    max_contour = np.concatenate([max_contour[left_id:, :], max_contour[:left_id, :]])
    # 计算中心与手部轮廓的距离, 根据 fft 频域分析得到关键 (凹凸) 点 (指尖, 连接处)
    dist = np.sqrt(np.square(max_contour - [x_c, y_c]).sum(-1))
    freq = np.fft.rfft(dist)
    cutoff = 15
    freq_new = np.concatenate([freq[:cutoff], 0 * freq[cutoff:]])
    dist_cut = np.fft.irfft(freq_new)
    """# 频域图
    plt.figure(figsize=(15, 5))
    plt.grid()
    plt.plot(dist, label='Original ED function', color='r', linewidth='3', linestyle='--')
    plt.plot(dist_cut, label='Low frequency reconstruction', color='b', linestyle='-')
    plt.legend()
    plt.show()"""
    derivative = np.diff(dist_cut)  # 差分
    sign_change = np.diff(np.sign(derivative)) / 2  # 变号处即为凹凸点
    """# 差分, 符号图
    plt.figure(figsize=(15, 5))
    plt.plot(sign_change)
    plt.plot(derivative)
    plt.axhline(y=0, color='r')
    plt.grid()
    plt.show()"""
    # 提取关键点
    minima = max_contour[np.where(sign_change > 0)[0]]
    v1, v2 = minima[-1], minima[-3]
    # 可视化其位置
    show(img)
    plt.plot(v1[0], v1[1], 'rx')
    plt.plot(v2[0], v2[1], 'bx')
    plt.show()
    # 消除手部旋转
    theta = np.arctan2((v2 - v1)[1], (v2 - v1)[0]) * 180 / np.pi
    # print('The rotation of ROI is {:.02f}\u00b0'.format(theta))
    R = cv.getRotationMatrix2D(v2.astype(np.float64), theta, 1)
    rotate_img = cv.warpAffine(img, R, (w, h))
    v1 = (R[:, :2] @ v1 + R[:, -1]).astype(np.uint)
    v2 = (R[:, :2] @ v2 + R[:, -1]).astype(np.uint)
    """# 查看校正图像
    plt.plot(v1[0], v1[1], 'rx')
    plt.plot(v2[0], v2[1], 'bx')
    show(rotate_img)"""
    plt.show()
    # 计算 ROI 区域
    ux = v1[0]
    uy = v1[1] + (v2 - v1)[0] // 3
    lx = v2[0]
    ly = v2[1] + 4 * (v2 - v1)[0] // 3
    """cv.rectangle(rotate_img, (lx, ly), (ux, uy), (0, 255, 0), 2)
    show(rotate_img)
    plt.show()"""
    roi = rotate_img[uy:ly, ux:lx, :]
    """plt.imshow(roi)
    plt.show()"""
    return roi
