import numpy as np
import training_config


class RGBConverter(object):
    def __init__(self):
        pass

    def __call__(self, img):
        # convert to RGB : stack 3 identical planes
        img = np.stack((img,) * 3, axis=1)

        # normalize for image-net weights
        for i in range(img.shape[0]):
            # for each plane
            for j in range(img.shape[1]):
                # for each color
                img[i][j] = (img[i][j] - training_config.IMAGENET_MEAN[j]) / training_config.IMAGENET_STDDEV[j]

        return img
