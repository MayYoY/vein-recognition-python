import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)]) + 0.001
    data[data == 0] = min_nonzero
    return data


def SSR(img, size=(3, 3), sigma=0., show=False):
    """
    单尺度 Retinex
    :param img:
    :param size:
    :param sigma:
    :param show:
    :return:
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if show:
        ax = plt.subplot(121)
        ax.set_title("raw image")
        plt.imshow(gray, cmap="gray")
    L_blur = cv.GaussianBlur(gray, size, sigma)
    gray = replaceZeroes(gray)
    L_blur = replaceZeroes(L_blur)

    dst = cv.log(gray / 255.0)
    dst_Lblur = cv.log(L_blur / 255.0)
    dst_IxL = cv.multiply(dst, dst_Lblur)
    log_R = cv.subtract(dst, dst_IxL)

    dst_R = cv.normalize(log_R, None, 0, 255, cv.NORM_MINMAX)
    dst_R = cv.convertScaleAbs(dst_R)
    if show:
        ax = plt.subplot(122)
        ax.set_title("enhanced image")
        plt.imshow(dst_R, cmap="gray")
        plt.show()
    dst_R = cv.cvtColor(dst_R, cv.COLOR_GRAY2BGR)
    return dst_R


def MSR(img, scales=None, show=False):
    if scales is None:
        scales = [3, 5, 9]
    number = len(scales)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if show:
        ax = plt.subplot(121)
        ax.set_title("raw image")
        plt.imshow(gray, cmap="gray")

    h, w = img.shape[:2]
    dst_R = np.zeros((h, w), dtype=np.float32)
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(number):
        gray = replaceZeroes(gray)
        L_blur = cv.GaussianBlur(gray, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        # log_R += np.log(gray + 0.001) - np.log(L_blur + 0.001)
        dst_img = cv.log(gray / 255.0)
        dst_Lblur = cv.log(L_blur / 255.0)
        dst_Ixl = cv.multiply(dst_img, dst_Lblur)
        log_R += cv.subtract(dst_img, dst_Ixl)

    log_R = log_R / number
    cv.normalize(log_R, dst_R, 0, 255, cv.NORM_MINMAX)
    dst_R = cv.convertScaleAbs(dst_R)
    if show:
        ax = plt.subplot(122)
        ax.set_title("enhanced image")
        plt.imshow(dst_R, cmap="gray")
        plt.show()
    dst = cv.cvtColor(dst_R, cv.COLOR_GRAY2BGR)
    # dst = cv.add(img, dst)

    return dst
