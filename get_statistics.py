import stain_utils

import numpy as np
from scipy.stats import pearsonr
import cv2 as cv

from os import listdir, mkdir
from os.path import isfile, join, isdir

def get_pearson_corr(I1, I2):
    # print("Original-", I1.shape, I2.shape)
    if I1.shape[0] != I2.shape[0] or I1.shape[1] != I2.shape[1]:
        I2 = cv.resize(I2, (I1.shape[1], I1.shape[0]))
    # print("Scaled-", I1.shape, I2.shape)
    I1 = np.sum(I1, axis=2) // 3
    I2 = np.sum(I2, axis=2) // 3
    I1 = I1.reshape((I1.shape[0]*I1.shape[1]))
    I2 = I2.reshape((I2.shape[0]*I2.shape[1]))
    # r = (np.average(I1*I2) - np.average(I1) * np.average(I2)) / (np.std(I1) * np.std(I2))
    return pearsonr(I1, I2)

# paths
output_path = "/home/fred/Projects/srt-cancer-img/pair-data/output/"
ground_path = "/home/fred/Projects/srt-cancer-img/pair-data/"
suffix = "frames/x40/"

methods = ['reinhard', 'macenko', 'vahadane']
for method in methods:
    imgs = [f for f in listdir(join(output_path, method)) if isfile(join(output_path, method, f))]
    pearson = []
    for img in imgs:
        # print(join(output_path, method, img))
        output = stain_utils.read_image(join(output_path, method, img))
        ground_truth_path = join(ground_path, img.split('_')[0].replace('A', 'H', 1), suffix, img.replace('A', 'H', 1))
        # print(ground_truth_path)
        ground = stain_utils.read_image(ground_truth_path)
        # print("Ground truth img: ", ground_truth_path)

        # pearson
        r = get_pearson_corr(output, ground)
        pearson.append(r)
    
    print(np.average(pearson))