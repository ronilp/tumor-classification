import os
import classification_config


def get_patient_id(npz_path):
    return npz_path.split("/")[-1].replace(".npz", "")


def get_plane_at_index(image_arr_entry):
    details = image_arr_entry.split(":")
    class_name = details[0]
    patient_id = details[1]
    plane = details[2]
    return (class_name, patient_id, int(plane))


def join_paths(npz_path, class_name, patient_id):
    return os.path.join(os.path.join(npz_path, class_name), patient_id)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if
               os.path.isdir(os.path.join(dir, d)) and d in classification_config.ALLOWED_CLASSES]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
