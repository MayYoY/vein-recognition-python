import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from scipy.spatial.distance import cosine
import d2l.torch as d2l
import matplotlib.pyplot as plt
from torch.utils import data
from . import utils, veinnet
from configs import running

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_net(num_class: int) -> nn.Module:
    net = torchvision.models.resnet18()
    del net.fc
    net.add_module("fc", nn.Linear(512, num_class))
    # net.add_module("fc", nn.Linear(2048, num_class))

    def xavier(module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d:
            nn.init.xavier_normal_(module.weight)
    net.apply(xavier)
    return net


def train_test(config):
    train_set = utils.ImageDataset(config.train_path, enhancement=config.enhance,
                                   transform=config.train_trans)
    train_iter = data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)

    test_set = utils.ImageDataset(config.test_path, enhancement=config.enhance,
                                  transform=config.test_trans)
    test_iter = data.DataLoader(test_set, batch_size=1, shuffle=False)

    # net = get_net(len(train_set.classes))
    net = veinnet.VeinNet(len(train_set.classes))
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr
                                 , weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config.num_epochs)
    loss_fun = nn.CrossEntropyLoss()

    # training
    utils.train(net, train_iter, optimizer, scheduler, loss_fun, config.num_epochs, config.device)
    # test
    utils.test(net, test_iter, config.device)
    if config.save:
        assert config.name, "saved name is required!"
        torch.save(net.state_dict(), f"./saved/{config.name}.pt")


def infer(config: running.NeuralInfer):
    """
    :param config:
        config.feature: 是否可视化特征
        config.probability: 预测概率
        config.compare: 比较类内, 类间
    :return:
    """
    test_set = utils.ImageDataset(config.test_path, enhancement=config.enhance,
                                  transform=config.test_trans)
    net = get_net(len(test_set.classes))
    net.load_state_dict(torch.load(config.model_path))
    net.eval()

    # 选自己的手指
    # class2 6, 7
    path, _ = test_set.samples[7]  # path, target
    img = cv.imread(path)
    img = cv.cvtColor(cv.resize(img, (224, 224)), cv.COLOR_BGR2RGB)
    img = np.asarray(img).astype(np.float32) / 255.
    x, y = test_set[7]
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    pred = net(x)
    print(pred.argmax(), y[0, 0])
    if config.feature:
        draw_cam(net, img, x, y)
    if config.probability:
        temp = pred.reshape(1, 1, 1, -1)
        d2l.show_heatmaps(temp, xlabel="classes", ylabel="samples")
        plt.show()
    if config.compare:
        sim = [0.] * len(test_set.class_to_idx)
        cnt = [0] * len(sim)
        for i in range(len(test_set)):
            if i == 7:  # class2 6 / 7
                continue
            x, y = test_set[i]
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            cnt[y[0, 0]] += 1
            temp = net(x)[0]
            # 余弦距离, 皮尔逊相关系数 or others
            sim[y[0, 0]] += cosine(pred[0].detach().numpy(), temp.detach().numpy())
            # sim[y[0, 0]] += np.corrcoef(pred[0].detach().numpy(), temp.detach().numpy())[0, 1]
        sim = [a / b for a, b in zip(sim, cnt)]
        plt.plot(range(len(sim)), sim)
        plt.title("Cosine Distance (target = 3)")
        # plt.title("Pearson Correlation Coefficient (target = 3)")
        plt.xlabel("Classes")
        plt.show()


def draw_cam(net: nn.Module, img: np.ndarray, x: torch.tensor, y: torch.tensor):
    """
    :param net: resnet18
    :param img: RGB
    :param x:
    :param y:
    :return:
    """
    target_layers = [net.layer4[-1]]
    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
    targets = [ClassifierOutputTarget(y[0])]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=x, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    ax = plt.subplot(121)
    ax.set_title("feature map")
    plt.imshow(visualization)
    ax = plt.subplot(122)
    ax.set_title("resized image")
    plt.imshow(img)
    plt.show()
