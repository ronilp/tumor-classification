import training_config
from utils.constants import CROP_CENTER, CROP_BOTTOM


class Cropper(object):
    def __init__(self):
        pass

    def __call__(self, img):
        pad = int((img.shape[1] - training_config.CROP_SIZE) / 2)
        if training_config.CROP_MID_POINT == CROP_CENTER:
            cropped_img = img[:, pad:-pad, pad:-pad]
        elif training_config.CROP_MID_POINT == CROP_BOTTOM:
            cropped_img = img[:, img.shape[1] - training_config.CROP_SIZE:img.shape[1], pad:-pad]
        else:
            # No crop
            cropped_img = img

        return cropped_img
