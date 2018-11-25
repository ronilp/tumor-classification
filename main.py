import copy
import multiprocessing
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
from tqdm import tqdm

from classification_config import GPU_MODE, CUDA_DEVICE, DATA_DIR, BATCH_SIZE, NUM_CLASSES
from mri_2d_dataset import MRI_2D_Dataset
from training_utils import exp_lr_scheduler

if GPU_MODE:
    torch.cuda.set_device(CUDA_DEVICE)

datasets = {x: MRI_2D_Dataset(os.path.join(DATA_DIR, x)) for x in ['train', 'val']}
dataset_loaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers=multiprocessing.cpu_count()) for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

MODEL_PREFIX = "dipg_vs_normals"


def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=5):
    since = time.time()

    best_model = model
    best_acc = 0.0

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
            running_corrects = 0

            # Training batch loop
            for data in tqdm(dataset_loaders[phase]):
                inputs, labels = data

                if GPU_MODE:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())
                    except Exception as e:
                        print("Exception while moving to cuda :", inputs, labels)
                        print(str(e))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backprop
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                try:
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)
                    # print(labels.data)
                    # print(preds)
                    # print('accuracy :', float(running_corrects)/preds.shape[0])
                except Exception as e:
                    print('Exception in calculating loss :' + str(e))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / float(dataset_sizes[phase])
            print('\nEpoch: {} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, phase, epoch_loss, epoch_acc))
            time_elapsed = time.time() - since
            print('Time Elapsed :{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            # Save best model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)
                    print('Best accuracy: {:4f}'.format(best_acc))
                    model_name = MODEL_PREFIX + "_" + str(epoch) + "_" + str(time.time()) + ".pt"
                    torch.save(model_ft.state_dict(), model_name)
                    print("Saved model :", model_name)

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val accuracy: {:4f}'.format(best_acc))
    return best_model


model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

criterion = nn.CrossEntropyLoss()

if GPU_MODE:
    criterion.cuda()
    model_ft.cuda()

optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=0.0001)

# Run the functions and save the best model in the function model_ft.
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=1)

print("Training done")

# Save model
model_name = MODEL_PREFIX + "_final_" + str(time.time()) + ".pt"
torch.save(model_ft.state_dict(), model_name)
print("Saved model :", model_name)
