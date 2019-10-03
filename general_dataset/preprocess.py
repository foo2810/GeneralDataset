# Preprocessing utils

import cv2
from PIL import Image

import numpy as np

# その他オーグメンションメソッドを用意

class Rescale():
    def __init__(self, rescaleRate):
        self.rescaleRate = rescaleRate
    
    def __call__(self, img):
        return img * self.rescaleRate

class Shift():
    def __init__(self, wShiftRange, hShiftRange):
        self.wShiftRange = wShiftRange
        self.hShiftRange = hShiftRange

    def __call__(self, img):
        w = img.shape[1] * self.wShiftRange / 2. * (2 * np.random.rand() - 1)
        h = img.shape[0] * self.hShiftRange / 2. * (2 * np.random.rand() - 1)
        M = np.float32([[1, 0, w],
                        [0, 1, h]])
        row, col, ch = img.shape
        trans = cv2.warpAffine(img, M, (col, row), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        return trans
