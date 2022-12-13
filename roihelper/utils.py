from . import finger, palm
import os
import glob
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm.auto import tqdm


def process(config):
    first_paths = glob.glob(config.input_path + os.sep + "*")
    subjects = os.listdir(config.input_path + os.sep)
    bar = tqdm(range(len(first_paths)))
    for i, fp in enumerate(first_paths):
        second_paths = glob.glob(fp + os.sep + "*.bmp")
        indexes = os.listdir(fp + os.sep)
        for j, sp in enumerate(second_paths):
            src = cv.imread(sp)
            if config.mode == "finger":
                dst = finger.getROI(src)
                if len(dst) == 0:
                    dst = finger.getROI(src, 10.)
            else:
                dst = palm.getROI(src)
            if not len(dst):
                continue
            save_dir = config.output_path + os.sep + subjects[i] + os.sep
            os.makedirs(save_dir, exist_ok=True)
            print(save_dir + indexes[j])
            cv.imwrite(save_dir + indexes[j], dst)
        bar.update(1)


def compare(subject):
    raw_path = ""
    roi_path = ""
    raw = cv.imread(raw_path, 0)
    roi = cv.imread(roi_path)
    ax = plt.subplot(121)
    ax.set_title("raw image")
    ax.imshow(raw, cmap="gray")
    ax = plt.subplot(122)
    ax.set_title("roi")
    roi = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
    ax.imshow(roi)
    plt.show()
