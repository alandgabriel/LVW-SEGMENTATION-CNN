#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 22:50:36 2021

@author: alan
"""

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import glob
import cv2
import random
import os


def gaussianShadow (img):
    img = img.astype(float)
    size = min(np.shape(img) [0], np.shape(img) [1])
    ksize =  int(size * (.1 + (.9-.1) * random.random()))
    sigma =  ksize / 4
    s = .4 + (.9 - .4) * random.random()
    gauss = cv2.getGaussianKernel(ksize, sigma)
    gauss = gauss / np.max(gauss)
    gauss = 1 - s * (gauss*gauss.T)
    shadowI = img.copy()
    rx = random.randrange(0, np.shape(img)[0] - ksize)
    ry = random.randrange(0, np.shape(img) [1] - ksize)
    shadowI [rx: rx + ksize, ry: ry + ksize] *= gauss 
    return shadowI

def rotateImg(img, mask):
    rAngle = random.randrange(-20, 20) 
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, rAngle, 1.0)
    imgR = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    maskR = cv2.warpAffine(mask, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return imgR, maskR

def addSpeckleNoise (img):
    rn = random.random() * .3
    rows,cols = np.shape(img)
    gauss =  rn *np.random.randn(rows,cols)       
    noisy = img + img * gauss
    return noisy

imgSize = (112,112)
folder = 'training'
extension = '.mhd'
files = []
for r, d, f in os.walk(folder):
    for file in f:
        if file.endswith(extension):
            files.append(os.path.join(r, file))
#AUGMENTATION OF DATA
images = [s for s in files if ('sequence' not in s) and ('gt' not in s)] 
masks = [s for s in files if ('sequence' not in s) and ('gt' in s)] 
for i in range(len(images)):
    img = np.squeeze(io.imread(images[i], plugin='simpleitk'))
    mask = np.squeeze (io.imread(masks[i], plugin='simpleitk'))
    mask [(mask==3) + (mask ==1)] = 0
    mask [mask==2] = 255
    mask = cv2.resize(mask, imgSize, interpolation = cv2.INTER_AREA)
    for j in range(30):
        img = gaussianShadow(img)
        #img, mask = rotateImg(img, mask)
        img = addSpeckleNoise(img)
        img = cv2.resize(img, imgSize, interpolation = cv2.INTER_AREA)
        cv2.imwrite('augmentedData/dat{}.png'.format(i*20 + j), img)
        cv2.imwrite('masks/dat{}.png'.format(i*20 + j), mask)

