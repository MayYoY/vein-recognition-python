import cv2 as cv
import numpy as np


def globalEqualHist(img):
    """
    全局直方图均衡化
    :param img:
    :return:
    """
    # 如果想要对图片做均衡化，必须将图片转换为灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)  # 在说明文档中有相关的注释与例子
    dst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    return dst


def CLAHE(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    dst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    return dst
