import numpy as np
import training_config


class IntensityNormalizer(object):
    def __init__(self):
        pass

    def __call__(self, img, manufacturer, scanner):
        intensity = training_config.INTENSITY_DICT[manufacturer][scanner]
        max = np.float(intensity[training_config.MAX])
        mean = np.float(intensity[training_config.MEAN])
        stddev = np.float(intensity[training_config.STD_DEV])

        # z-score normalization
        if training_config.INTENSITY_Z_SCORE_NORM:
            img = (img - mean) / stddev

        # bring to [0, 1] scale
        if training_config.INTENSITY_01_NORM:
            img = np.float32(img / max)
            mask = (img > 1.0)
            img[mask] = 1.0

        # min-max normalization : bring to [0, 255] scale
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * training_config.MAX_PIXEL_VAL

        return img
