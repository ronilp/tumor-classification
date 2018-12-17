import random

from training_config import CROP_SIZE


class AugmentedImageScaler(object):
    """
        Sample image by taking alternate pixels.
    """

    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img: 3D image tensor of the form planes x height x width to be sampled from.

        Returns:
            3D image: One of the four possible randomly sampled images.
        """
        # No sampling if original image size < 2 * CROP_SIZE
        if img.shape[1] < 2 * CROP_SIZE or img.shape[2] < 2 * CROP_SIZE:
            return img

        prob = random.random()

        if prob <= 0.25:
            return img[:, ::2, ::2]  # ee
        elif prob <= 0.5:
            return img[:, 1::2, 1::2]  # oo
        elif prob <= 0.75:
            return img[:, ::2, 1::2]  # eo
        else:
            return img[:, 1::2, ::2]  # oo

    def __repr__(self):
        return self.__class__.__name__
