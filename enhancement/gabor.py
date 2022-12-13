import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import pylab as pl
from PIL import Image


# 构建gabor滤波器
def buildGaborFilter():
    filters = []
    kernel_size = [7, 9]  # gabor尺度
    # sigma = [100,50]
    # lambda_val = np.pi/4.0  # 波长
    lambda_val = 20
    for k in range(len(kernel_size)):  # 0
        for theta in np.arange(0, np.pi, np.pi / 8):  # gabor方向
            kern = cv.getGaborKernel((kernel_size[k], kernel_size[k]),
                                     100, theta, lambda_val, 1, 0, ktype=cv.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)  # 8个filter
    plt.figure(1)

    """# 用于绘制滤波器
    for temp in range(len(filters)):  # (0,7)
        plt.subplot(4, 4, temp + 1)
        plt.imshow(filters[temp])
    plt.show()"""
    return filters


# Gabor增强
def gaborProcess(img, filters):  # gray img
    res = []  # 滤波结果
    for i in range(len(filters)):  # temp (0,7)`
        # res1 = process(img, filters[i])
        acc = np.zeros_like(img)
        for kern in filters[i]:
            filtered_img = cv.filter2D(img, cv.CV_8UC1, kern)
            acc = np.maximum(acc, filtered_img, acc)
        res.append(np.array(acc))  # the result img, the size of the res is the same as img

    # 用于绘制滤波效果
    img2 = np.zeros_like(img)
    img3 = np.zeros_like(img)

    """plt.figure(2)
    for temp in range(len(res)):  # temp:(0,7)
        plt.subplot(4, 4, temp + 1)
        plt.imshow(res[temp], cmap='gray')
    plt.show()"""

    h, w = img.shape[0:2]
    for row in range(h):
        for col in range(w):
            res_pixel = []
            res_pixel2 = []
            for single_res in range(8):  # single res is an img in res   # (0,7)
                res_pixel.append(res[single_res][row, col])  # append one pixel in each gabor_img
                res_pixel2.append(res[single_res + 8][row, col])
                # print('res_pixel ',res_pixel)
                # min_res_pixel = min(res_pixel)
            # medium_res_pixel = np.median(res_pixel)
            # avg_res_pixel = np.mean(res_pixel)
            """max_res_pixel = max(res_pixel)
            max_res_pixel2 = max(res_pixel2)
            img2[row, col] = max_res_pixel
            img3[row, col] = max_res_pixel2"""
            res_pixel = np.asarray(res_pixel)
            res_pixel2 = np.asarray(res_pixel2)
            max_res_pixel = res_pixel.max()
            max_res_pixel2 = res_pixel2.max()
            img2[row, col] = max_res_pixel
            img3[row, col] = max_res_pixel2

    # img2 = res[1]
    result_img = [img2, img3]
    return result_img  # 返回滤波结果


def build_filters():
    filters = []
    ksize = [7, 9, 11, 13, 15, 17]  # gabor尺度 6个
    lamda = np.pi / 2.0  # 波长

    for theta in np.arange(0, np.pi, np.pi / 4):  # gabor方向 0 45 90 135
        for k in range(6):
            kern = cv.getGaborKernel((ksize[k], ksize[k]), 1.0, theta, lamda, 0.5, 0, ktype=cv.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


# 滤波过程
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv.filter2D(img, cv.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


# 特征图生成并显示
def getGabor(path, filters):
    image = Image.open(path)
    img_ndarray = np.asarray(image)
    res = []  # 滤波结果
    for i in range(len(filters)):
        res1 = process(img_ndarray, filters[i])
        res.append(np.asarray(res1))

    pl.figure(2)
    for temp in range(len(res)):
        pl.subplot(4, 6, temp + 1)  # 画4*6格子
        pl.imshow(res[temp], cmap='gray')
    pl.show()

    return res


if __name__ == "__main__":
    path = ""
    img = cv.imread(path)
    gabor_filters = buildGaborFilter()  # 构建gabor滤波器
    results = gaborProcess(img, gabor_filters)  # Gabor增强原图像

    filters = build_filters()
    getGabor(path, filters)
