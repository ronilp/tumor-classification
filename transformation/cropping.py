import training_config
from utils.constants import CROP_CENTER, CROP_BOTTOM
import numpy as np


class Cropper(object):
    def __init__(self):
        pass

    def __call__(self, img):
        pad = int((min(img.shape[1], img.shape[2]) - training_config.CROP_SIZE) / 2)
        if training_config.CROP_MID_POINT == CROP_CENTER:
            cropped_img = img[:, pad:-pad, pad:-pad]
        elif training_config.CROP_MID_POINT == CROP_BOTTOM:
            if pad != 0:
                cropped_img = img[:, img.shape[1] - training_config.CROP_SIZE:img.shape[1], pad:-pad]
            else:
                cropped_img = img[:, img.shape[1] - training_config.CROP_SIZE:img.shape[1], :]
        else:
            # No crop
            cropped_img = img

        return cropped_img
