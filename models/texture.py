import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from tqdm.auto import tqdm
import random
import joblib

from enhancement import hist_method, retinex, stretch
from . import utils
from configs import running


def textureMatch(config):
    train_set = utils.ImageDataset(config.train_path, enhancement=None)
    test_set = utils.ImageDataset(config.test_path, enhancement=None)
    path = "./saved/" + config.name
    if config.train:
        # training
        getRecord(train_set, save_path=path, enhance=config.enhance, method=config.method)
    # testing
    test(test_set, model_path=path, enhance=config.enhance, method=config.method)


def imfilter(img, kernel):
    kernel_flip = cv.flip(kernel, 0)
    anchor = (int(kernel.shape[1] - kernel.shape[1] / 2 - 1),
              int(kernel.shape[0] - kernel.shape[0] / 2 - 1))
    res = cv.filter2D(img, -1, kernel_flip, anchor=anchor, delta=0,
                      borderType=cv.BORDER_REPLICATE)
    return res


def maxCurvature(raw, mask=None, enhance=None, sigma=2., show=False):
    """
    :param raw:
    :param mask:
    :param enhance: 不增强很难提取纹理
    :param sigma:
    :param show: 是否可视化
    :return:
    """
    if enhance is None:
        enhance = hist_method.CLAHE
    if mask is None:
        mask = np.ones((raw.shape[0], raw.shape[1]))
    # img = enhance(raw)
    img = raw.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    win_size = np.ceil(4 * sigma)
    x = np.arange(-win_size, win_size + 1)
    y = np.arange(-win_size, win_size + 1)
    X, Y = np.meshgrid(x, y)

    # filter parameters
    h = (1 / (2 * pi * sigma ** 2)) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    hx = (-X / (sigma ** 2)) * h
    hxx = ((X ** 2 - sigma ** 2) / (sigma ** 4)) * h
    hy = hx.T
    hyy = hxx.T
    hxy = ((X * Y) / (sigma ** 4)) * h
    # filter
    fx = imfilter(gray, hx)  # -_conv(src, hx)
    fxx = imfilter(gray, hxx)
    fy = imfilter(gray, hy)
    fyy = imfilter(gray, hyy)
    fxy = imfilter(gray, hxy)  # -_conv(src, hxy)
    f1 = 0.5 * np.sqrt(2) * (fx + fy)  # \  #
    f2 = 0.5 * np.sqrt(2) * (fx - fy)  # /  #
    f11 = 0.5 * fxx + fxy + 0.5 * fyy  # \\ #
    f22 = 0.5 * fxx - fxy + 0.5 * fyy  # // #

    # 逐方向处理
    h, w = gray.shape
    k = np.zeros((h, w, 4))
    k[:, :, 0] = (fxx / ((1 + fx ** 2) ** (3 / 2))) * mask  # horizontal #
    k[:, :, 1] = (fyy / ((1 + fy ** 2) ** (3 / 2))) * mask  # vertical #
    k[:, :, 2] = (f11 / ((1 + f1 ** 2) ** (3 / 2))) * mask  # \   #
    k[:, :, 3] = (f22 / ((1 + f2 ** 2) ** (3 / 2))) * mask  # /   #
    # Scores
    Vt = np.zeros_like(gray)
    Wr = 0
    # Horizontal direction
    bla = k[:, :, 0] > 0
    for y in range(h):
        for x in range(w):
            if bla[y, x]:
                Wr += 1
            if Wr > 0 and (x == (w - 1) or not bla[y, x]):
                if x == (w - 1):
                    # Reached edge of image
                    pos_end = x
                else:
                    pos_end = x - 1

                pos_start = pos_end - Wr + 1  # Start pos of concave
                if pos_start == pos_end:
                    temp = k[y, pos_start, 0].argmax()
                else:
                    temp = k[y, pos_start:pos_end + 1, 0].argmax()

                pos_max = pos_start + temp
                Scr = k[y, pos_max, 0] * Wr
                Vt[y, pos_max] = Vt[y, pos_max] + Scr
                Wr = 0
    # Vertical direction
    bla = k[:, :, 1] > 0
    for x in range(w):
        for y in range(h):
            if bla[y, x]:
                Wr = Wr + 1
            if Wr > 0 and (y == (h - 1) or not bla[y, x]):
                if y == (h - 1):
                    # Reached edge of image
                    pos_end = y
                else:
                    pos_end = y - 1

                pos_start = pos_end - Wr + 1  # Start pos of concave
                if pos_start == pos_end:
                    temp = k[pos_start, x, 1].argmax()
                else:
                    temp = k[pos_start:pos_end + 1, x, 1].argmax()

                pos_max = pos_start + temp
                Scr = k[pos_max, x, 1] * Wr
                Vt[pos_max, x] = Vt[pos_max, x] + Scr
                Wr = 0
    # Direction: \
    bla = k[:, :, 2] > 0
    for start in range(0, w + h - 1):
        # Initial values
        if start <= w - 1:
            x = start
            y = 0
        else:
            x = 0
            y = start - w + 1
        done = False

        while not done:
            if bla[y, x]:
                Wr = Wr + 1

            if Wr > 0 and (y == h - 1 or x == w - 1 or not bla[y, x]):
                if y == h - 1 or x == w - 1:
                    # Reached edge of image
                    pos_x_end = x
                    pos_y_end = y
                else:
                    pos_x_end = x - 1
                    pos_y_end = y - 1

                pos_x_start = pos_x_end - Wr + 1
                pos_y_start = pos_y_end - Wr + 1

                if pos_y_start == pos_y_end and pos_x_start == pos_x_end:
                    d = k[pos_y_start, pos_x_start, 2]
                elif pos_y_start == pos_y_end:
                    d = np.diag(k[pos_y_start, pos_x_start:pos_x_end + 1, 2])
                elif pos_x_start == pos_x_end:
                    d = np.diag(k[pos_y_start:pos_y_end + 1, pos_x_start, 2])
                else:
                    d = np.diag(k[pos_y_start:pos_y_end + 1, pos_x_start:pos_x_end + 1, 2])

                temp = d.argmax()

                pos_x_max = pos_x_start + temp
                pos_y_max = pos_y_start + temp

                Scr = k[pos_y_max, pos_x_max, 2] * Wr

                Vt[pos_y_max, pos_x_max] = Vt[pos_y_max, pos_x_max] + Scr
                Wr = 0

            if (x == w - 1) or (y == h - 1):
                done = True
            else:
                x = x + 1
                y = y + 1
    # Direction: /
    bla = k[:, :, 3] > 0
    for start in range(0, w + h - 1):
        # Initial values
        if start <= (w - 1):
            x = start
            y = h - 1
        else:
            x = 0
            y = w + h - start - 1
        done = False

        while not done:
            if bla[y, x]:
                Wr = Wr + 1
            if Wr > 0 and (y == 0 or x == w - 1 or not bla[y, x]):
                if y == 0 or x == w - 1:
                    # Reached edge of image
                    pos_x_end = x
                    pos_y_end = y
                else:
                    pos_x_end = x - 1
                    pos_y_end = y + 1

                pos_x_start = pos_x_end - Wr + 1
                pos_y_start = pos_y_end + Wr - 1

                if pos_y_start == pos_y_end and pos_x_start == pos_x_end:
                    d = k[pos_y_end, pos_x_start, 3]
                elif pos_y_start == pos_y_end:
                    d = np.diag(np.flipud(k[pos_y_end, pos_x_start:pos_x_end + 1, 3]))
                elif pos_x_start == pos_x_end:
                    d = np.diag(np.flipud(k[pos_y_end:pos_y_start + 1, pos_x_start, 3]))
                else:
                    d = np.diag(np.flipud(k[pos_y_end:pos_y_start + 1, pos_x_start:pos_x_end + 1, 3]))

                temp = d.argmax()
                pos_x_max = pos_x_start + temp
                pos_y_max = pos_y_start - temp
                Scr = k[pos_y_max, pos_x_max, 3] * Wr
                Vt[pos_y_max, pos_x_max] = Vt[pos_y_max, pos_x_max] + Scr
                Wr = 0

            if (x == w - 1) or (y == 0):
                done = True
            else:
                x = x + 1
                y = y - 1

    # Connection of vein centres
    Cd = np.zeros((h, w, 4))
    for x in range(2, w - 3):
        for y in range(2, h - 3):
            Cd[y, x, 0] = min(Vt[y, x + 1:x + 3].max(), Vt[y, x - 2:x].max())
            Cd[y, x, 1] = min(Vt[y + 1:y + 3, x].max(), Vt[y - 2:y, x].max())
            Cd[y, x, 2] = min(Vt[y - 2:y, x - 2:x].max(), Vt[y + 1:y + 3, x + 1:x + 3].max())
            Cd[y, x, 3] = min(Vt[y + 1:y + 3, x - 2:x].max(), Vt[y - 2:y, x + 1:x + 3].max())

    # Veins
    img_veins = Cd.max(axis=2)
    _, img_veins_bin = cv.threshold(img_veins, 0, 255, cv.THRESH_BINARY)
    if show:
        ax = plt.subplot(131)
        ax.set_title("vein texture")
        ax.imshow(img_veins_bin, cmap="gray")
        ax = plt.subplot(132)
        ax.set_title("enhanced image (CLAHE)")
        ax.imshow(gray, cmap="gray")
        ax = plt.subplot(133)
        ax.set_title("raw image")
        gray_raw = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
        ax.imshow(gray_raw, cmap="gray")
        plt.suptitle("Maximum Curvature", fontsize=15, x=0.5, y=0.8)
        plt.show()
    return img_veins_bin


def repeatedLineTrack(raw, mask=None, enhance=None, iterations=15000,
                      r=10, W=19, show=False):
    if enhance is None:
        enhance = hist_method.CLAHE
    if mask is None:
        mask = np.ones((raw.shape[0], raw.shape[1]))
    img = enhance(raw)
    # img = raw.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = (gray / 255.).astype(np.float64)
    height, width = gray.shape

    p_lr = 0.5
    p_ud = 0.25
    Tr = np.zeros_like(gray, dtype=np.uint8)
    bla = np.array([[-1, -1], [-1, 0], [-1, 1],
                    [0, -1], [0, 0], [0, 1],
                    [1, -1], [1, 0], [1, 1]])

    assert W % 2, "W must be even!"
    ro = round(r * np.sqrt(2) / 2)
    hW = int((W - 1) / 2)
    hWo = round(hW * np.sqrt(2) / 2)
    # 忽略不可达区域
    for y in range(width):
        for x in range(int(r + hW + 1)):
            x = min(height - 1, x)
            y = min(width - 1, y)
            mask[x, y] = 0
            mask[height - x - 1, y] = 0
    for x in range(height):
        for y in range(int(r + hW + 1)):
            x = min(height - 1, x)
            y = min(width - 1, y)
            mask[x, y] = 0
            mask[x, width - y - 1] = 0
    indices = []
    # 随机生成初始点, x 横坐标, y 纵坐标
    for _ in range(iterations):
        y = random.randint(0, width - 1)  # xRandom = rng.uniform(0, mask.cols)
        x = random.randint(0, height - 1)  # yRandom = rng.uniform(0, mask.rows)
        if mask[x, y]:
            indices.append(np.array([x, y]))
    # 开始迭代
    for start_pt in indices:
        xc = start_pt[0]
        yc = start_pt[1]
        # 决定运动方向
        if random.randint(0, 1):
            Dlr = -1
        else:
            Dlr = 1
        if random.randint(0, 1):
            Dud = -1
        else:
            Dud = 1
        # Tc = cv::Mat::zeros(src.size(), CV_8U);
        Tc = np.zeros_like(gray, dtype=np.uint8)
        Vl = 1
        while Vl > 0:
            # cv::Mat Nr = cv::Mat::zeros(cv::Size(3, 3), CV_8U);
            Nr = np.zeros((3, 3), dtype=np.uint8)
            p = random.uniform(0, 1)
            if p < p_lr:
                # go left or right
                Nr[:, 1 + Dlr] = 1
            elif p_lr < p < p_lr + p_ud:
                # go up or down
                Nr[1 + Dud, :] = 1
            else:
                # any direction
                Nr[:, :] = 1
                Nr[1, 1] = 0
            Nc = []
            # 搜寻候选点
            for dx in range(-1, 2):  # [-1, 1]
                for dy in range(-1, 2):
                    x = xc + dx
                    y = yc + dy
                    if not Tc[x, y] and Nr[dx + 1, dy + 1] and mask[x, y]:
                        temp = (dx + 1) * 3 + (dy + 1)
                        Nc.append(np.array([xc + bla[temp, 0], yc + bla[temp, 1]]))
            if not Nc:
                break
            # std::vector<double> Vdepths(Nc.size());
            Vdepths = np.zeros(len(Nc))
            for i, Ncp in enumerate(Nc):
                if Ncp[1] == yc:  # vertical
                    yp = yc
                    if Ncp[0] > xc:
                        xp = Ncp[0] + r
                    else:
                        xp = Ncp[0] - r
                    Vdepths[i] = gray[xp, yp + hW] - 2 * gray[xp, yp] + gray[xp, yp - hW]
                elif Ncp[0] == xc:  # horizontal
                    xp = xc
                    if Ncp[1] > yc:
                        yp = Ncp[1] + r
                    else:
                        yp = Ncp[1] - r
                    Vdepths[i] = gray[xp + hW, yp] - 2 * gray[xp, yp] + gray[xp - hW, yp]
                elif Ncp[0] > xc and Ncp[1] < yc or Ncp[0] < xc and Ncp[1] > yc:
                    if Ncp[0] > xc and Ncp[1] < yc:  # bottom left
                        xp = Ncp[0] + ro
                        yp = Ncp[1] - ro
                    else:  # top right
                        xp = Ncp[0] - ro
                        yp = Ncp[1] + ro
                    Vdepths[i] = gray[xp - hWo, yp - hWo] - 2 * gray[xp, yp] \
                                 + gray[xp + hWo, yp + hWo]
                else:  # Diagonal
                    if Ncp[0] < xc and Ncp[1] < yc:  # top left
                        xp = Ncp[0] - ro
                        yp = Ncp[1] - ro
                    else:
                        xp = Ncp[0] + ro
                        yp = Ncp[1] + ro
                    Vdepths[i] = gray[xp + hWo, yp - hWo] - 2 * gray[xp, yp] \
                                 + gray[xp - hWo, yp + hWo]

                a = 1
                Tc[xc, yc] = 1  # 标记探索过
                Tr[xc, yc] += 1
                idx = Vdepths.argmax()
                xc, yc = Nc[idx][0], Nc[idx][1]
                Vl = Vdepths[idx]

    if show:
        ax = plt.subplot(131)
        ax.set_title("vein texture")
        ax.imshow(Tr, cmap="gray")
        ax = plt.subplot(132)
        ax.set_title("enhanced image (CLAHE)")
        ax.imshow(gray, cmap="gray")
        ax = plt.subplot(133)
        ax.set_title("raw image")
        gray_raw = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
        ax.imshow(gray_raw, cmap="gray")
        plt.suptitle("Repeated Line Tracking", fontsize=15, x=0.5, y=0.8)
        plt.show()
    return Tr


def getRecord(train_set: utils.ImageDataset, save_path, method="repeatedLineTrack", enhance=None):
    assert method in ["repeatedLineTrack", "maxCurvature"]
    print("Training")
    record_texture = []
    record_y = []
    bar = tqdm(range(len(train_set)))
    for path, target in train_set.samples:
        raw = cv.imread(path)
        if method == "repeatedLineTrack":
            temp = repeatedLineTrack(raw, enhance=enhance)
        else:
            temp = maxCurvature(raw, sigma=2., enhance=enhance)
        record_texture.append(temp)
        record_y.append(target)
        bar.update(1)
    joblib.dump((record_texture, record_y), save_path, compress=4)  # 压缩省空间


def test(test_set: utils.ImageDataset, model_path, method="repeatedLineTrack", enhance=None):
    assert method in ["repeatedLineTrack", "maxCurvature"]
    print("Testing")
    record_texture, record_y = joblib.load(model_path)
    accuracy = 0
    bar = tqdm(range(len(test_set)))
    for path, target in test_set.samples:
        raw = cv.imread(path)
        if method == "repeatedLineTrack":
            texture = repeatedLineTrack(raw, enhance=enhance)
        else:
            texture = maxCurvature(raw, sigma=2., enhance=enhance)
        texture = (texture / 255.).astype(np.float64)
        pred = -1  # 预测标签
        score = -1  # 最佳匹配值
        for i, rt in enumerate(record_texture):
            rt = cv.resize(rt, (texture.shape[1], texture.shape[0]))
            rt = (rt / 255.).astype(np.float64)
            temp = cv.bitwise_and(texture, rt).sum() / cv.bitwise_or(texture, rt).sum()
            if temp > score:
                score = temp
                pred = record_y[i]
        if pred == target:
            accuracy += 1
        bar.update(1)
    print(f"Accuracy: {accuracy / len(test_set)}")


def show(config: running.TextureShow):
    """
    随机选取测试样本, 展示 LBP 特征以及直方图
    :param config:
        config.test_set:
        config.model_path:
    :return:
    """
    test_set = utils.ImageDataset(config.test_path, enhancement=config.enhance)
    record_hist, record_y = joblib.load(config.model_path)
    path, target = test_set.samples[21]
    raw = cv.imread(path)
    if config.method == "repeatedLineTrack":
        texture = repeatedLineTrack(raw, enhance=config.enhance)
    else:
        texture = maxCurvature(raw, sigma=2., enhance=config.enhance)
    texture = (texture / 255.).astype(np.float64)

    avg_score = [0.] * len(test_set.classes)
    cnt = [0] * len(avg_score)
    scores = {"data": [], "label": []}
    pred = -1  # 预测标签
    score = -1  # 最佳匹配值
    for i, rt in enumerate(record_hist):
        rt = cv.resize(rt, (texture.shape[1], texture.shape[0]))
        rt = (rt / 255.).astype(np.float64)
        temp = cv.bitwise_and(texture, rt).sum() / cv.bitwise_or(texture, rt).sum()

        # 计算平均距离
        avg_score[record_y[i]] += temp
        cnt[record_y[i]] += 1
        scores["data"].append(temp)
        if record_y[i] != target:
            scores["label"].append("outer")
        else:
            scores["label"].append("inner")
        if temp > score:
            score = temp
            pred = record_y[i]
    print(target, pred)
    avg_score = [a / b for a, b in zip(avg_score, cnt)]
    plt.plot(range(len(avg_score)), avg_score)
    plt.title("Average Score (target = 10)")
    plt.xlabel("Classes")
    plt.show()

    scores = pd.DataFrame(scores)

    ax = sns.displot(data=scores, x="data", hue="label", kind="kde")
    ax.set_titles("Score Distribution")
    plt.show()


def compareIter(img):
    ret = repeatedLineTrack(img, iterations=5000, W=19)
    ax = plt.subplot(221)
    ax.set_title("iterations = 3000")
    ax.imshow(ret, cmap="gray")
    ret = repeatedLineTrack(img, iterations=10000, W=19)
    ax = plt.subplot(222)
    ax.set_title("iterations = 10000")
    ax.imshow(ret, cmap="gray")
    ret = repeatedLineTrack(img, iterations=15000, W=19)
    ax = plt.subplot(223)
    ax.set_title("iterations = 15000")
    ax.imshow(ret, cmap="gray")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ax = plt.subplot(224)
    ax.set_title("raw image")
    ax.imshow(gray, cmap="gray")
    plt.show()
