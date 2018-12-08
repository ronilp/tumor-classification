import csv
import os
import pickle
from multiprocessing import Pool
import logging
import numpy as np
import pydicom as dicom
logging.basicConfig(level=logging.DEBUG)

FILEPATH_DICT_PATH = "/Users/rpancholia/Documents/Acads/Projects/data/classification-pkl"
ROOT_SRC_PATH = "/Users/rpancholia/Documents/Acads/Projects/data/classification-dcm/"
PREPROCESS_DEST_PATH = "/Users/rpancholia/Documents/Acads/Projects/data/classification-pkl/"
CLASSES = ["DIPG", "MB", "EP"]


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
        patient_id = splitted_terms[1]
        study_id = "-".join(splitted_terms[2:-1])
        key = patient_id + "_" + study_id
        if key not in patient_study_vs_files.keys():
            patient_study_vs_files[key] = []

        patient_study_vs_files[key].append(os.path.join(source_dir, file_name))
    dump_to_csv(patient_study_vs_files, os.path.join(FILEPATH_DICT_PATH, class_name),
                headers=["patient_study_id", "files"])
    logging.info("Created dictionary for patient Ids vs file paths: " + str(class_name))
    return patient_study_vs_files


# patient_study_vs_files = populate_filepath_dict(class_name)

def stack_images(patient_study_id):
    """ Create pixel array pickles from raw DICOM images """
    global class_name, patient_study_vs_files
    logging.info(patient_study_id + " " + class_name)
    try:
        files = patient_study_vs_files[patient_study_id]
        stacked_image_planes = []
        for file_path in files:
            with dicom.read_file(file_path) as dcm:
                plane = np.array(dcm.pixel_array)
                plane = np.squeeze(plane)
                stacked_image_planes.append(plane)

        img_npy = np.float16(np.array(stacked_image_planes))

        if img_npy.shape[0] == len(stacked_image_planes):
            # to make sure that the third index ranges up to len(stacked_image_planes)
            img_npy = np.swapaxes(img_npy, 0, 2)
            img_npy = np.swapaxes(img_npy, 0, 1)

        # make first index len(stacked_image_planes)
        img_npy = np.swapaxes(img_npy, 0, 2)
        img_npy = np.swapaxes(img_npy, 1, 2)

        dest_file = os.path.join(os.path.join(PREPROCESS_DEST_PATH, class_name), patient_study_id)
        with open(dest_file + ".pkl", "wb") as pickle_file:
            pickle.dump(img_npy, pickle_file)
            # print ("Saved pickle for {}".format(patient_study_id))
    except Exception as e:
        logging.error("Exception in processing " + str(patient_study_id))
        logging.error(e)


if __name__ == "__main__":
    init_dirs()

    for class_name in CLASSES:
        patient_study_vs_files = populate_filepath_dict(class_name)
        pool = Pool()
        _ = pool.map(stack_images, patient_study_vs_files.keys())
