import numpy as np
import pandas as pd
import cv2 as cv
import joblib
from tqdm.auto import tqdm
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import itemfreq
import matplotlib.pyplot as plt
import seaborn as sns

from configs import running
from . import utils


def lbpMatch(config):
    train_set = utils.ImageDataset(config.train_path, enhancement=None)
    test_set = utils.ImageDataset(config.test_path, enhancement=None)
    path = "./saved/" + config.name
    if config.train:
        # training
        getRecord(train_set, save_path=path, enhance=config.enhance)
    # testing
    test(test_set, path, enhance=config.enhance)


def getRecord(train_set: utils.ImageDataset, save_path, enhance=None):
    print("Training")
    record_hist = []
    record_y = []
    bar = tqdm(range(len(train_set)))
    for path, target in train_set.samples:
        img = cv.imread(path)
        if enhance is not None:
            img = enhance(img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转为灰度图
        feature, hist = getLBP(gray)
        """# 可视化 LBP 特征 (编码图)
        plt.imshow(feature, cmap="gray")
        plt.show()"""
        record_hist.append(hist)
        record_y.append(target)
        bar.update(1)
    joblib.dump((record_hist, record_y), save_path, compress=4)  # 压缩省空间


def test(test_set, model_path, enhance=None):
    print("Testing")
    record_hist, record_y = joblib.load(model_path)
    accuracy = 0
    bar = tqdm(range(len(test_set)))
    for path, target in test_set.samples:
        img = cv.imread(path)
        if enhance is not None:
            img = enhance(img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        feature, hist = getLBP(gray)
        pred = -1  # 预测标签
        score = 1e5  # 最佳匹配值
        for i, rh in enumerate(record_hist):
            # cv.HISTCMP_CORREL 相关性比较, 分数越高越匹配
            # cv.HISTCMP_CHISQR 卡方比较, 分数越低越匹配
            # cv.HISTCMP_BHATTACHARYYA 巴氏距离比较, 分数越低越匹配
            T = min(len(hist), len(rh))
            temp = cv.compareHist(hist[: T].astype(np.float32), rh[: T].astype(np.float32),
                                  method=cv.HISTCMP_CHISQR)
            if temp < score:
                score = temp
                pred = record_y[i]
        if pred == target:
            accuracy += 1
        bar.update(1)
    print(f"Accuracy: {accuracy / len(test_set)}")


def getLBP(gray):
    r = 3  # LBP 算法的邻域半径
    p = 8 * r  # 8 邻域
    feature = local_binary_pattern(gray, p, r, method='uniform')  # LBP 特征提取
    freq = itemfreq(feature.ravel())  # 频率统计
    hist = freq[:, 1] / sum(freq[:, 1])  # 得到直方图
    return feature, hist


def show(config: running.LBPShow):
    """
    随机选取测试样本, 展示 LBP 特征以及直方图
    :param config:
        config.test_set:
        config.model_path:
    :return:
    """
    test_set = utils.ImageDataset(config.test_path, enhancement=config.enhance)
    record_hist, record_y = joblib.load(config.model_path)
    # class2: light1 6 / 7
    path, target = test_set.samples[21]
    img = cv.imread(path)
    if config.enhance is not None:
        img = config.enhance(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    feature, hist = getLBP(gray)
    ax = plt.subplot(121)
    ax.imshow(feature, cmap="gray")
    ax.set_title("LBP Feature")
    ax = plt.subplot(122)
    ax.imshow(gray, cmap="gray")
    ax.set_title("Raw Image")
    plt.show()

    avg_dist = [0.] * len(test_set.classes)
    cnt = [0] * len(avg_dist)
    dists = {"data": [], "label": []}
    pred = -1  # 预测标签
    score = 1e5  # 最佳匹配值
    for i, rh in enumerate(record_hist):
        T = min(len(hist), len(rh))
        temp = cv.compareHist(hist[: T].astype(np.float32), rh[: T].astype(np.float32),
                              method=cv.HISTCMP_CHISQR)
        # 计算平均距离
        avg_dist[record_y[i]] += temp
        cnt[record_y[i]] += 1
        dists["data"].append(temp)
        if record_y[i] != target:
            dists["label"].append("outer")
        else:
            dists["label"].append("inner")
        if temp < score:
            score = temp
            pred = record_y[i]
    print(target, pred)
    avg_dist = [a / b for a, b in zip(avg_dist, cnt)]
    plt.plot(range(len(avg_dist)), avg_dist)
    plt.title("Average Distance (target = 3)")
    plt.xlabel("Classes")
    plt.show()

    # 归一化距离, 方便展示分布
    scaler = MinMaxScaler()
    dists["data"] = scaler.fit_transform(np.asarray(dists["data"]).reshape(-1, 1)).reshape(-1)
    dists = pd.DataFrame(dists)

    ax = sns.displot(data=dists, x="data", hue="label", kind="kde")
    ax.set_titles("Distance Distribution")
    plt.show()
