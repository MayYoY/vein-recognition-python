import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def grayNormalize(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    max_val = gray.max()
    min_val = gray.min()
    dst = (gray - min_val) * (255 / (max_val - min_val))

    """# histogram of low contrast image
    pre = [0] * 256
    post = [0] * 256
    for i, row in enumerate(gray):
        for j, val in enumerate(row):
            pre[val] += 1
            post[int(dst[i, j])] += 1
    plt.plot(pre, color="b")
    plt.plot(post, color="r")
    plt.show()"""

    dst = cv.cvtColor(dst.astype(np.uint8), cv.COLOR_GRAY2BGR)
    return dst
