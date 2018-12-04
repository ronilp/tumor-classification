import torch
from torch.autograd import Variable
from torchvision import models
from tqdm import tqdm

from dataset_utils import load_testset
from MRI_Dataset.mri_2d_classification_dataset import MRI_2D_Classification_Dataset
from training_config import GPU_MODE
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

model = models.resnet152()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

model.load_state_dict(torch.load('checkpoints/dipg_vs_mb_289_1543417080.7.pt', map_location='cpu'))

dataset_loaders, dataset_sizes = load_testset(MRI_2D_Classification_Dataset)

running_corrects = 0

y_pred = []
y_true = []

for data in tqdm(dataset_loaders['test']):
    inputs, labels = data

    if GPU_MODE:
        try:
            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
        except Exception as e:
            print("Exception while moving to cuda :", inputs, labels)
            print(str(e))
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    outputs = model(inputs)
    vals, preds = torch.max(outputs.data, 1)
    running_corrects += torch.sum(preds == labels.data)

    for x in preds:
        y_pred.append(x)

    for x in labels:
        y_true.append(x)

target_names = ["DIPG", "MB"]
print(classification_report(y_true, y_pred, target_names=target_names))
print('accuracy :', float(running_corrects) / dataset_sizes['test'])
print(f1_score(y_true, y_pred, average="macro"))
print(precision_score(y_true, y_pred, average="macro"))
print(recall_score(y_true, y_pred, average="macro"))