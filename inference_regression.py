import torch
from torch.autograd import Variable
from torchvision import models
from tqdm import tqdm

from dataset_utils import load_testset
from MRI_Dataset.mri_2d_regression_dataset import MRI_2D_Regression_Dataset
from training_config import GPU_MODE
from training_utils import regression_metrics

model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)

model.load_state_dict(torch.load('age_regression_4_1543152855.63.pt', map_location='cpu'))

dataset_loaders, dataset_sizes = load_testset(MRI_2D_Regression_Dataset)

running_mse = 0.0
running_r2 = 0.0

y_pred = []
y_true = []

count = 0
for data in tqdm(dataset_loaders['test']):
    count += 1
    print (str(count))
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
    y_pred_sum = 0.0
    for x in outputs.detach().numpy():
        y_pred_sum += x

    y_pred.append(y_pred_sum/15.0)

    y_true_sum = 0.0
    for x in labels.detach().numpy():
        y_true_sum += x

    y_true.append(y_true_sum / 15.0)

mse, r2_score = regression_metrics(y_true, y_pred)
print('mse: {} r2_score: {}'.format(mse, r2_score))

# mse: 67.86006927490234 r2_score: 0.8269445281378675