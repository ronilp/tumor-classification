import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

import training_config
from utils import dataset_utils


class MRI_3D_Classification_Dataset(Dataset):
    def __init__(self, npz_path, mode, transforms=None):
        # Read the file path
        self.data_path = npz_path

        # Set transformers
        self.transforms = transforms

        # Read class names
        self.classes, self.class_to_idx = dataset_utils.find_classes(self.data_path)

        # Read image_arr and label_arr from npz
        self.image_arr = []
        self.label_arr = []
        class_counts = {}
        for class_name in training_config.ALLOWED_CLASSES:
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                continue

            subject_count = 0
            for patient_id_npz in os.listdir(class_path):
                subject_count += 1
                if subject_count > training_config.CLASS_LIMIT:
                    break
                if not patient_id_npz.endswith(".npz"):
                    continue

                path = dataset_utils.join_paths(self.data_path, class_name, patient_id_npz)
                self.image_arr.append(path)
                self.label_arr.append(self.class_to_idx[class_name])

            class_counts[class_name + "_subjects"] = subject_count

        # Calculate len
        self.data_len = len(self.image_arr)

        print(mode + " subject counts : " + str(class_counts))

    def __getitem__(self, index):
        npz_path = self.image_arr[index]

        npz = np.load(npz_path)
        filters = list(npz.keys())
        filter = filters[0]
        raw_img = npz.get(filter)
        npz.close()

        pad = int((raw_img.shape[1] - 244) / 2)
        raw_img = raw_img[pad:-pad, pad:-pad, :]

        raw_img = np.swapaxes(raw_img, 0, 2)
        raw_img = np.swapaxes(raw_img, 1, 2)

        # standardize
        raw_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img)) * training_config.MAX_PIXEL_VAL

        # convert to RGB
        raw_img = np.stack((raw_img,) * 3, axis=1)

        if self.transforms is not None:
            raw_img = self.transforms(raw_img)

        # Transform image to tensor
        img_as_tensor = torch.Tensor(raw_img)

        # Get label of the image
        image_label = torch.tensor(self.label_arr[index])

        return (img_as_tensor, image_label)

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    custom_dataset = MRI_3D_Classification_Dataset(os.path.join(training_config.DATA_DIR, 'train'), 'train')
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
