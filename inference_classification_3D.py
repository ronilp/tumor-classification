import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from torch.autograd import Variable
from tqdm import tqdm

from models.MRNet_2D import MRNet_2D
from models.MRNet import MRNet
from mri_dataset.mri_3d_pkl_dataset import MRI_3D_PKL_Dataset
from mri_dataset.mri_3d_transformer_dataset import MRI_3D_Transformer_Dataset
from training_config import GPU_MODE, NUM_CLASSES
from transformation.aug_rescaler import AugmentedImageScaler
from transformation.cropping import Cropper
from transformation.rgb_converter import RGBConverter
from utils.dataset_utils import load_testset_from_csv
from torchvision import transforms

model = MRNet_2D(NUM_CLASSES)
model.load_state_dict(torch.load('dipg_vs_mb_vs_eb_checkpoints/dipg_vs_mb_vs_eb_1_1545103167.473354.pt', map_location='cpu'))

transforms = transforms.Compose([
        AugmentedImageScaler(),
        Cropper(),
        RGBConverter()
    ])

dataset_loaders, dataset_sizes, datasets = load_testset_from_csv(MRI_3D_Transformer_Dataset, transforms)


# invert class_to_idx
y_pred_dict = {}
class_to_idx = datasets['test'].class_to_idx
idx_to_class = {}
for key in class_to_idx.keys():
    idx_to_class[class_to_idx[key]] = key
    y_pred_dict[key] = []

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

    # outputs[0] since batch size always = 1
    for i, log_prob in enumerate(outputs[0]):
        y_pred_dict[idx_to_class[i]].append(log_prob)

    for x in preds:
        y_pred.append(x)

    for x in labels:
        y_true.append(x)


target_names = []
for key, value in sorted(idx_to_class.items()):
    target_names.append(value)

print(classification_report(y_true, y_pred, target_names=target_names))
print('accuracy :', float(running_corrects) / dataset_sizes['test'])
print("overall f1 weighted:", f1_score(y_true, y_pred, average="weighted"))
print("overall precision weighted:", precision_score(y_true, y_pred, average="weighted"))
print("overall recall weighted:", recall_score(y_true, y_pred, average="weighted"))

images = datasets['test'].image_arr
labels = datasets['test'].label_arr

# Converting tensors to values
y_true_val = []
y_pred_val = []
y_pred_MB = []
y_pred_EP = []
y_pred_DIPG = []
image_names = []
for i in range(len(y_true)):
    y_true_val.append(y_true[i].item())
    y_pred_val.append(y_pred[i].item())
    y_pred_MB.append(y_pred_dict['MB'][i].item())
    y_pred_EP.append(y_pred_dict['EP'][i].item())
    y_pred_DIPG.append(y_pred_dict['DIPG'][i].item())
    # get image name from absolute path
    # Can modify this to get patient id/study id/something better
    image_names.append(images[i].split('/')[-1])

results_df = pd.DataFrame(
    {'image_path': image_names,
     'label': y_true_val,
     'prediction': y_pred_val,
     'log_probablity_mb': y_pred_MB,
     'log_probablity_ep': y_pred_EP,
     'log_probablity_dipg': y_pred_DIPG
    })

results_df.to_csv("results.csv", index=False)