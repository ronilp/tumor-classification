import multiprocessing
import os

import torch

import training_config
from utils.constants import GE, SIEMENS, SCANNER_15T, SCANNER_3T, PHILIPS


def get_patient_id(npz_path):
    return npz_path.split("/")[-1].replace(".npz", "")


def get_plane_at_index(image_arr_entry):
    details = image_arr_entry.split(":")
    class_name = details[0]
    patient_id = details[1]
    filter = details[2]
    plane = details[3]
    return (class_name, patient_id, filter, int(plane))


def join_paths(npz_path, class_name, patient_id):
    return os.path.join(os.path.join(npz_path, class_name), patient_id)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if
               os.path.isdir(os.path.join(dir, d)) and d in training_config.ALLOWED_CLASSES]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def load_datasets(Dataset_Class):
    datasets = {x: Dataset_Class(os.path.join(training_config.DATA_DIR, x), x) for x in ['train', 'val']}
    dataset_loaders = {
    x: torch.utils.data.DataLoader(datasets[x], batch_size=training_config.BATCH_SIZE, shuffle=True,
                                   num_workers=multiprocessing.cpu_count()) for x in
    ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    return dataset_loaders, dataset_sizes

def load_testset(Dataset_Class):
    datasets = {x: Dataset_Class(os.path.join(training_config.DATA_DIR, x), x) for x in ['test']}
    dataset_loaders = {
    x: torch.utils.data.DataLoader(datasets[x], batch_size=training_config.BATCH_SIZE, shuffle=True,
                                   num_workers=multiprocessing.cpu_count()) for x in ['test']}
    dataset_sizes = {x: len(datasets[x]) for x in ['test']}
    return dataset_loaders, dataset_sizes


def interleave_images(pixel_array):
    ee = pixel_array[::2, ::2]
    oo = pixel_array[1::2, 1::2]
    eo = pixel_array[::2, 1::2]
    oe = pixel_array[1::2, ::2]
    return ee, oo, eo, oe

def get_manufacturer(dcm):
    if "GE MEDICAL SYSTEMS" == dcm.Manufacturer:
        manufacturer = GE
    elif dcm.Manufacturer.startswith("Philips"):
        manufacturer = PHILIPS
    else:
        manufacturer = SIEMENS
    return manufacturer


def get_scanner(dcm):
    if dcm.MagneticFieldStrength == 3:
        scanner = SCANNER_3T
    else:
        scanner = SCANNER_15T
    return scanner