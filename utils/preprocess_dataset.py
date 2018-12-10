import sys
import csv
import logging
import os
import pickle
from multiprocessing.pool import ThreadPool

import numpy as np
import pydicom as dicom
sys.path.append("..")
from utils.constants import GE, SCANNER_15T
from utils.dataset_utils import interleave_images, get_manufacturer, get_scanner

logging.basicConfig(level=logging.DEBUG)

FILEPATH_DICT_PATH = "/Users/rpancholia/Documents/Acads/Projects/data/classification-pkl"
ROOT_SRC_PATH = "/Users/rpancholia/Documents/Acads/Projects/data/classification-dcm/"
PREPROCESS_DEST_PATH = "/Users/rpancholia/Documents/Acads/Projects/data/classification-pkl-augmented/"
CLASSES = ["DIPG", "MB", "EP"]
CROP_SIZE = 224

# Flag to augment images by sampling pixels
SAMPLE_IMAGES = True

intensity_dict = {}

def create_missing_dirs(path):
    if not os.path.exists(path):
        logging.info("Creating directory at " + path)
        os.makedirs(path)


def init_dirs():
    create_missing_dirs(PREPROCESS_DEST_PATH)

    for class_name in CLASSES:
        create_missing_dirs(os.path.join(PREPROCESS_DEST_PATH, class_name))


# Populate patient/study vs file paths dictionary
def dump_to_csv(my_dict, path, headers=None):
    with open(path + ".csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        if headers is not None:
            writer.writerow(headers)
        for key, value in my_dict.items():
            writer.writerow([key, value])


def populate_filepath_dict(class_name):
    patient_study_vs_files = {}
    source_dir = os.path.join(ROOT_SRC_PATH, class_name)
    file_list = os.listdir(source_dir)
    file_list.sort()
    for file_name in file_list:
        if not file_name.endswith(".dcm"):
            continue
        splitted_terms = file_name.split("-")
        patient_id = splitted_terms[1].replace("_", "") # remove _ from patient_ids
        study_id = "-".join(splitted_terms[2:-1])
        key = patient_id + "_" + study_id
        if key not in patient_study_vs_files.keys():
            patient_study_vs_files[key] = []

        patient_study_vs_files[key].append(os.path.join(source_dir, file_name))
    dump_to_csv(patient_study_vs_files, os.path.join(PREPROCESS_DEST_PATH, class_name),
                headers=["patient_study_id", "files"])
    logging.info("Created dictionary for patient Ids vs file paths: " + str(class_name))
    return patient_study_vs_files


# patient_study_vs_files = populate_filepath_dict(class_name)


def fix_dims(img_npy):
    """ Make sure that the third index ranges up to len(stacked_image_planes) """
    img_npy = np.swapaxes(img_npy, 0, 2)
    img_npy = np.swapaxes(img_npy, 0, 1)
    return img_npy


def rearrange_axes(img_npy):
    """ Make first index len(stacked_image_planes) """
    img_npy = np.swapaxes(img_npy, 0, 2)
    img_npy = np.swapaxes(img_npy, 1, 2)
    return img_npy


def handle_dims(img_npy, stacked_image_planes):
    if img_npy.shape[0] == len(stacked_image_planes):
        fix_dims(img_npy)

    return rearrange_axes(img_npy)


def save_npy(img_npy, patient_study_id, manufacturer, scanner, sample=""):
    global intensity_dict
    dest_file = os.path.join(os.path.join(PREPROCESS_DEST_PATH, class_name), patient_study_id + sample)
    intensity_dict[dest_file + ".pkl"] = manufacturer + "_" + scanner
    with open(dest_file + ".pkl", "wb") as pickle_file:
        pickle.dump(img_npy, pickle_file)
        # print ("Saved pickle for {}".format(dest_file))


def stack_images(patient_study_id):
    """ Create pixel array pickles from raw DICOM images """
    global class_name, patient_study_vs_files
    logging.info(patient_study_id + " " + class_name)
    try:
        files = patient_study_vs_files[patient_study_id]
        stacked_image_planes = []
        ee_stacked_image_planes = []
        oo_stacked_image_planes = []
        eo_stacked_image_planes = []
        oe_stacked_image_planes = []
        b_augmenting = False

        # default manufacturer GE, default scanner 1.5T
        manufacturer = GE
        scanner = SCANNER_15T
        for file_path in files:
            with dicom.read_file(file_path) as dcm:
                try:
                    manufacturer = get_manufacturer(dcm)
                    scanner = get_scanner(dcm)
                except Exception as e:
                    logging.error("Exception identifying manufacturer or scanner:" + str(file_path))
                    logging.error(e)

                plane = np.array(dcm.pixel_array)
                if not SAMPLE_IMAGES or plane.shape[0] < 2 * CROP_SIZE or plane.shape[1] < 2 * CROP_SIZE:
                    stacked_image_planes.append(np.squeeze(plane))
                else:
                    b_augmenting = True
                    ee, oo, eo, oe = interleave_images(plane)
                    logging.info(patient_study_id + " " + class_name + " augmented planes")
                    ee_stacked_image_planes.append(np.squeeze(ee))
                    oo_stacked_image_planes.append(np.squeeze(oo))
                    eo_stacked_image_planes.append(np.squeeze(eo))
                    oe_stacked_image_planes.append(np.squeeze(oe))

        if not b_augmenting:
            img_npy = np.float16(np.array(stacked_image_planes))
            # img_npy = handle_dims(img_npy, stacked_image_planes)
            save_npy(img_npy, patient_study_id, manufacturer, scanner)
        else:
            plane_list = [ee_stacked_image_planes, oo_stacked_image_planes, oe_stacked_image_planes,
                          eo_stacked_image_planes]
            img_npy_list = [np.float16(np.array(stacked_image_planes)) for stacked_image_planes in plane_list]

            for i, img_npy in enumerate(img_npy_list):
                save_npy(img_npy, patient_study_id, manufacturer, scanner, "_" + str(i))
    except Exception as e:
        logging.error("Exception in processing " + str(patient_study_id))
        logging.error(e)


if __name__ == "__main__":
    init_dirs()
    for class_name in CLASSES:
        patient_study_vs_files = populate_filepath_dict(class_name)
        pool = ThreadPool()
        _ = pool.map(stack_images, patient_study_vs_files.keys())

    with open(os.path.join(PREPROCESS_DEST_PATH, "intensity_map.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["filename", "manufacturer", "scanner"])
        for key in intensity_dict.keys():
            values = intensity_dict[key].split("_")
            writer.writerow([key, values[0], values[1]])