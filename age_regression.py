import os
import copy
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from tqdm import tqdm

from dataset_utils import load_datasets
from mri_2d_regression_dataset import MRI_2D_Regression_Dataset
from training_config import GPU_MODE, CUDA_DEVICE, NUM_CLASSES, MODEL_PREFIX, BASE_LR
from training_utils import exp_lr_scheduler, regression_metrics

if GPU_MODE:
    torch.cuda.set_device(CUDA_DEVICE)

dataset_loaders, dataset_sizes = load_datasets(MRI_2D_Regression_Dataset)

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=5):
    since = time.time()

    best_model = model
    best_r2 = 0.0

    # Training loop
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_mse = 0.0
            running_r2 = 0.0

            y_pred = []
            y_true = []

            # Training batch loop
            count = 0
            for data in tqdm(dataset_loaders[phase]):
                count += 1
                inputs, labels = data

                if GPU_MODE:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                    except Exception as e:
                        print("Exception while moving to cuda :", inputs, labels)
                        print(str(e))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                preds = outputs
                loss = criterion(outputs, labels.detach())

                # backprop
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                try:
                    running_loss += loss.item()
                    mse, r2_score = regression_metrics(labels.detach().numpy(), preds.detach().numpy())
                    running_mse += mse
                    running_r2 += r2_score
                    # print("Target :", labels.data)
                    # print("Predicted :", preds.detach())
                    print('mse: {} r2_score: {}'.format(mse, r2_score))
                except Exception as e:
                    print('Exception in calculating regression metrics :' + str(e))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_mse = running_mse / count
            epoch_r2 = running_r2 / count
            print('\nEpoch: {} {} Loss: {:.4f} MSE: {:.4f} R2: {:.4f}'.format(epoch + 1, phase, epoch_loss, epoch_mse,
                                                                              epoch_r2))
            time_elapsed = time.time() - since
            print('Time Elapsed :{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            # Save best model
            if phase == 'val':
                if epoch_r2 > best_r2:
                    best_r2 = epoch_r2
                    best_model = copy.deepcopy(model)
                    print('Best accuracy: {:4f}'.format(best_r2))
                    model_name = MODEL_PREFIX + "_" + str(epoch) + "_" + str(time.time()) + ".pt"
                    torch.save(model_ft.state_dict(), os.path.join("checkpoints", model_name))
                    print("Saved model :", model_name)

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val accuracy: {:4f}'.format(best_r2))
    return best_model


model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# MSE for regression
criterion = nn.MSELoss()

if GPU_MODE:
    criterion.cuda()
    model_ft.cuda()

optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=BASE_LR)

# Run the functions and save the best model in the function model_ft.
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)

print("Training done")

# Save model
model_name = MODEL_PREFIX + "_final_" + str(time.time()) + ".pt"
torch.save(model_ft.state_dict(), os.path.join("checkpoints", model_name))
print("Saved model :", model_name)
