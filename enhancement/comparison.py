from . import hist_method, retinex, stretch
import cv2 as cv
import matplotlib.pyplot as plt


def show(img):

    r1 = hist_method.globalEqualHist(img)
    r1 = cv.cvtColor(r1, cv.COLOR_BGR2GRAY)
    ax = plt.subplot(231)
    ax.set_title("globalEqualHist")
    ax.imshow(r1, cmap="gray")

    r2 = hist_method.CLAHE(img)
    r2 = cv.cvtColor(r2, cv.COLOR_BGR2GRAY)
    ax = plt.subplot(232)
    ax.set_title("CLAHE")
    ax.imshow(r2, cmap="gray")

    r3 = retinex.SSR(img)
    r3 = cv.cvtColor(r3, cv.COLOR_BGR2GRAY)
    ax = plt.subplot(233)
    ax.set_title("retinex SSR")
    ax.imshow(r3, cmap="gray")

    r4 = retinex.MSR(img)
    r4 = cv.cvtColor(r4, cv.COLOR_BGR2GRAY)
    ax = plt.subplot(234)
    ax.set_title("retinex MSR")
    ax.imshow(r4, cmap="gray")

    r5 = stretch.grayNormalize(img)
    r5 = cv.cvtColor(r5, cv.COLOR_BGR2GRAY)
    ax = plt.subplot(235)
    ax.set_title("grayNormalize")
    ax.imshow(r5, cmap="gray")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ax = plt.subplot(236)
    ax.set_title("raw img")
    ax.imshow(gray, cmap="gray")
    plt.show()
