from roihelper import utils, palm
from enhancement import comparison
from models import resnet, LBP, texture
import models
from configs import preprocess, running
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import d2l.torch as d2l
import random


# palm.getROI(img)
# comparison.show(img)
# utils.compare(100)
# models.utils.trainTestSplit(preprocess.SplitFinger)
# models.utils.trainTestSplit(preprocess.SplitPalm)
# utils.process(preprocess.PrePalm)
# resnet.train_test(running.NeuralFinger())
# resnet.train_test(running.NeuralPalm())
# resnet.infer(running.NeuralInfer())

# LBP.lbpMatch(running.LBPFinger())
LBP.lbpMatch(running.LBPPalm())
# LBP.show(running.LBPShow())

# retinex.MSR(img, scales=[1, 3, 5], show=True)
# comparison.show(img)
# texture.textureMatch(running.TexturePalm)
# texture.show(running.TextureShow())
