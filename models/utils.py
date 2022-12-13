import os
import glob
import random
import shutil
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils import data


class ImageDataset(data.Dataset):
    def __init__(self, path, enhancement=None, transform=None, target_transform=None):
        super(ImageDataset, self).__init__()
        self.path = path
        self.classes, self.class_to_idx = self.get_classes()  # [str], dict
        self.samples = self.get_samples()  # [(path, int)]
        self.targets = [s[1] for s in self.samples]
        self.enhancement = enhancement
        self.transform = transform
        self.target_transform = target_transform

    def get_classes(self):
        classes = sorted(entry.name for entry in os.scandir(self.path) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {self.path}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def get_samples(self):
        samples = []
        for target_class in self.classes:
            target_dir = os.path.join(self.path, target_class)
            if not os.path.isdir(target_dir):
                continue
            fnames = glob.glob(target_dir + os.sep + "*.bmp")
            for fname in fnames:
                item = fname, self.class_to_idx[target_class]
                samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        x = cv.imread(path)
        if self.enhancement is not None:
            x = self.enhancement(x)
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        y = torch.LongTensor([y])
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


def trainTestSplit(config):
    """
    :param config
        config.train_path: 原始路径, 抽取部分图像后, 剩下部分即为训练集
        config.test_path: 测试集路径, 存储测试集
        config.test_size: 测试集大小
    :return:
    """
    first_paths = glob.glob(config.train_path + os.sep + "*")
    subjects = os.listdir(config.train_path + os.sep)
    bar = tqdm(range(len(first_paths)))
    for i, fp in enumerate(first_paths):  # 单个受试者的文件
        indexes = os.listdir(fp + os.sep)
        random.shuffle(indexes)
        N = round(config.test_size * len(indexes))
        save_dir = config.test_path + os.sep + subjects[i]
        os.makedirs(save_dir, exist_ok=True)
        # print(save_dir)
        for j in range(N):
            # print(fp + os.sep + indexes[j])
            shutil.move(fp + os.sep + indexes[j], save_dir)

        bar.update(1)


class Accumulate:
    def __init__(self, n):
        self.n = n
        self.cnt = [0] * n
        self.acc = [0] * n

    def update(self, val: list, n):
        if not isinstance(n, list):
            n = [n] * self.n
        if not isinstance(val, list):
            val = [val] * self.n
        self.cnt = [a + b for a, b in zip(self.cnt, n)]
        self.acc = [a + b for a, b in zip(self.acc, val)]

    def reset(self):
        self.cnt = [0] * self.n
        self.acc = [0] * self.n


def train(net: nn.Module, train_iter: data.DataLoader,
          optimizer: torch.optim.Optimizer, scheduler,
          loss_fun, num_epochs, device):
    net.to(device)
    net.train()
    train_loss = []
    train_acc = Accumulate(1)
    print("Training...")
    bar = tqdm(range(num_epochs * len(train_iter)))
    for epoch in range(num_epochs):
        temp = np.zeros(1)
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device).reshape(-1)
            preds = net(x)

            optimizer.zero_grad()
            loss = loss_fun(preds, y)
            loss.backward()
            optimizer.step()

            train_acc.update(calAcc(preds, y), len(y))
            temp += loss.cpu().detach().numpy()
            bar.update(1)
        train_loss.append(temp / len(train_iter))
        # scheduler.step()  # 忘记用了, 用的话可能要改超参数, 懒得调 :(
    print(f"Train Accuracy: {train_acc.acc[0] / train_acc.cnt[0]}")
    plt.plot(range(num_epochs), train_loss)
    plt.title("Train Loss")
    plt.show()


def test(net: nn.Module, test_iter: data.DataLoader, device):
    net.to(device)
    net.eval()
    test_acc = Accumulate(1)
    print("Testing...")
    bar = tqdm(range(len(test_iter)))
    for x, y in test_iter:
        x = x.to(device)
        y = y.to(device).reshape(-1)
        preds = net(x)
        test_acc.update(calAcc(preds, y), len(y))
        bar.update(1)
    print(f"Test Accuracy: {test_acc.acc[0] / test_acc.cnt[0]}")


def calAcc(preds: torch.Tensor, labels: torch.Tensor, percentage=False):
    preds = preds.cpu().detach().numpy().argmax(axis=-1)
    labels = labels.cpu().numpy()
    if percentage:
        return (preds == labels).sum() / len(labels)
    else:
        return (preds == labels).sum()
