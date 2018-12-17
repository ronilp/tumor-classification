import csv
import os
import pickle

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import training_config
from transformation.aug_rescaler import AugmentedImageScaler
from transformation.cropping import Cropper
from transformation.rgb_converter import RGBConverter
from utils import dataset_utils


class MRI_3D_Transformer_Dataset(Dataset):
    def __init__(self, dir_path, mode, transforms=None):
        # Read the file path
        self.data_path = dir_path

        # Set transforms
        self.transforms = transforms

        # Read class names
        self.classes, self.class_to_idx = dataset_utils.find_classes(self.data_path)

        intensity_map = {}
        with open(os.path.join(dir_path, "intensity_map.csv")) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csv_reader:
                intensity_map[row[0]] = row[1] + "_" + row[2]

        # Read image_arr and label_arr from csv files
        self.image_arr = []
        self.label_arr = []
        self.scanner_arr = []
        self.manufacturer_arr = []
        self.class_counts = {}
        for class_name in training_config.ALLOWED_CLASSES:
            self.class_counts[class_name] = 0

        with open(os.path.join(dir_path, mode + ".csv")) as csvfile:
            dataset_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in dataset_reader:
                if self.class_counts[row[1]] > training_config.CLASS_LIMIT:
                    # in case you wish to limit number of instances per class
                    continue
                self.image_arr.append(row[0])
                self.label_arr.append(self.class_to_idx[row[1]])
                intensity_data = intensity_map[row[0]].split("_")
                self.manufacturer_arr.append(intensity_data[0])
                self.scanner_arr.append(intensity_data[1])
                self.class_counts[row[1]] += 1

        # Calculate len
        self.data_len = len(self.image_arr)

        print(mode + " class counts : " + str(self.class_counts))

    def __getitem__(self, index):
        pkl_path = self.image_arr[index]

        raw_img = pickle.load(open(pkl_path, "rb"))

        intensity = training_config.INTENSITY_DICT[self.manufacturer_arr[index]][self.scanner_arr[index]]
        max = np.float(intensity[training_config.MAX])
        mean = np.float(intensity[training_config.MEAN])
        stddev = np.float(intensity[training_config.STD_DEV])

        # z-score normalization
        if training_config.INTENSITY_Z_SCORE_NORM:
            raw_img = (raw_img - mean) / stddev

        # bring to [0, 1] scale
        if training_config.INTENSITY_01_NORM:
            raw_img = np.float32(raw_img / max)
            mask = (raw_img > 1.0)
            raw_img[mask] = 1.0

        # min-max normalization : bring to [0, 255] scale
        raw_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img)) * training_config.MAX_PIXEL_VAL

        if self.transforms is not None:
            raw_img = self.transforms(raw_img)

        # Transform image to tensor
        img_as_tensor = torch.Tensor(raw_img)

        # Get label of the image
        image_label = torch.tensor(self.label_arr[index])

        return (img_as_tensor, image_label)

    def __len__(self):
        return self.data_len

    # TODO: get rid of the assumption that batch size = 1
    def penalize_loss(self, loss, labels):
        for key in self.class_to_idx:
            if self.class_to_idx[key] == labels[0]:
                class_name = key
                break

        return loss / self.class_counts[class_name]


if __name__ == '__main__':
    transforms = transforms.Compose([
        AugmentedImageScaler(),
        Cropper(),
        RGBConverter()
    ])
    custom_dataset = MRI_3D_Transformer_Dataset(training_config.DATA_DIR, 'train', transforms=transforms)
    print(custom_dataset.classes)
    print(custom_dataset.data_len)

    (img, label) = custom_dataset.__getitem__(42)
    print("Single image shape :", img.shape)
    print("Single image label :", label)

    mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=training_config.BATCH_SIZE,
                                                    shuffle=False)

    for images, labels in mn_dataset_loader:
        # Feed the data to the model
        print(images.shape)
        print(labels)

        for i in range(len(images)):
            print(images[i].numpy().shape)
            print(labels[i])
