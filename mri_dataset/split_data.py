import os
import csv
import sys
from sklearn.model_selection import train_test_split
sys.path.append("..")
from training_config import RANDOM_SEED

DATA_DIR = "/Users/rpancholia/Documents/Acads/Projects/data/classification-pkl-augmented/"
CLASSES = ["DIPG", "MB", "EP"]


def stratified_split(X, y, test_size=0.2, validate_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size,
                                                        random_state=random_state)
    new_validate_size = validate_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=new_validate_size,
                                                      random_state=random_state)

    return X_train, X_test, X_val, y_train, y_test, y_val


def populate(X, Z, y):
    y_mod = []
    X_list = []
    for i, key in enumerate(Z):
        for file_name in X[key]:
            X_list.append(file_name)
            y_mod.append(y[i])
    return X_list, y_mod


def create_dataset(X, y, file_name):
    with open(os.path.join(DATA_DIR, file_name), 'w', newline='') as csvfile:
        dataset_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(X)):
            dataset_writer.writerow((X[i], y[i]))


if __name__ == "__main__":
    X = {}
    y = []
    Z = []
    for class_name in CLASSES:
        class_path = os.path.join(DATA_DIR, class_name)
        files = []
        for file_name in os.listdir(class_path):
            if not file_name.endswith(".pkl"):
                continue

            files.append(file_name)
            splitted_terms = file_name.split("_")
            patient_id = splitted_terms[0]
            value = class_name + "_" + patient_id

            if value not in X.keys():
                X[value] = []
                y.append(class_name)
                Z.append(value)

            X[value].append(os.path.join(class_path, file_name))

    Z_train, Z_test, Z_val, y_train, y_test, y_val = stratified_split(Z, y, test_size=0.2, validate_size=0.2,
                                                                      random_state=RANDOM_SEED)

    X_train, y_train = populate(X, Z_train, y_train)
    X_test, y_test = populate(X, Z_test, y_test)
    X_val, y_val = populate(X, Z_val, y_val)

    print("Train size: {}".format(len(X_train)))
    print("Test size: {}".format(len(X_test)))
    print("Val size: {}".format(len(X_val)))

    create_dataset(X_train, y_train, 'train.csv')
    create_dataset(X_test, y_test, 'test.csv')
    create_dataset(X_val, y_val, 'val.csv')
