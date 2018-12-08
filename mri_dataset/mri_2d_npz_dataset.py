import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

import training_config
from utils import dataset_utils


class MRI_2D_NPZ_Dataset(Dataset):
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
            plane_count = 0
            for patient_id_npz in os.listdir(class_path):
                subject_count += 1
                if subject_count > training_config.CLASS_LIMIT:
                    break
                if not patient_id_npz.endswith(".npz"):
                    continue

                path = dataset_utils.join_paths(self.data_path, class_name, patient_id_npz)
                npz = np.load(path)
                filters = list(npz.keys())
                for filter in filters:
                    patient_id = dataset_utils.get_patient_id(patient_id_npz)
                    for i in range(npz.get(filter).shape[2]):
                        plane_count += 1
                        self.image_arr.append(class_name + ":" + patient_id + ":" + filter + ":" + str(i))
                        self.label_arr.append(self.class_to_idx[class_name])

                npz.close()

            class_counts[class_name + "_subjects"] = subject_count
            class_counts[class_name + "_planes"] = plane_count

        # Calculate len
        self.data_len = len(self.image_arr)

        print(mode + " subject counts : " + str(class_counts))

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]

        # Open image
        (class_name, patient_id, filter, plane) = dataset_utils.get_plane_at_index(single_image_name)
        path = dataset_utils.join_paths(self.data_path, class_name, patient_id)

        npz = np.load(path + ".npz")
        raw_img = npz.get(filter)[:, :, plane]
        raw_img = np.resize(raw_img, (224, 224))

        # Expand dimension for a single channel-image
        if training_config.SINGLE_CHANNEL:
            raw_img = np.expand_dims(raw_img, axis=0)
        else:
            # Convert raw image to a 3 channel-image
            raw_img = np.stack((raw_img,) * 3, axis=0)

        if self.transforms is not None:
            raw_img = self.transforms(raw_img)
        npz.close()

        # Transform image to tensor
        img_as_tensor = torch.Tensor(raw_img)

        # Get label of the image
        single_image_label = torch.tensor(self.label_arr[index])

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    custom_dataset = MRI_2D_NPZ_Dataset(os.path.join(training_config.DATA_DIR, 'train'), 'train')
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
