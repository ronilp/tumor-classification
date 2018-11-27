import os
import copy
import sys
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
from tqdm import tqdm

from mri_2d_classification_dataset import MRI_2D_Classification_Dataset
from training_config import GPU_MODE, CUDA_DEVICE, NUM_CLASSES, MODEL_PREFIX, BASE_LR, BATCH_SIZE, LEARNING_PATIENCE
from dataset_utils import load_datasets
from training_utils import exp_lr_scheduler

if GPU_MODE:
    torch.cuda.set_device(CUDA_DEVICE)

dataset_loaders, dataset_sizes = load_datasets(MRI_2D_Classification_Dataset)

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=5):
    since = time.time()

    best_model = model
    best_acc = 0.0
    best_loss = sys.maxsize
    patience = 0

    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    early_stop = False

    # Training loop
    for epoch in range(num_epochs):
        if early_stop:
            break
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_processed = 0

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
                    total_processed += preds.shape[0]
                    # print(labels.data)
                    # print(preds)
                    # print('accuracy :', float(running_corrects) / total_processed)
                except Exception as e:
                    print('Exception in calculating loss :' + str(e))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / float(dataset_sizes[phase])
            print('\nEpoch: {} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, phase, epoch_loss, epoch_acc))
            time_elapsed = time.time() - since
            print('Time Elapsed :{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            if 'train' == phase:
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            # Save best model
            if phase == 'val':
                if epoch_loss < best_loss:
                    patience = 0
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(model)
                    print('Best accuracy: {:4f}'.format(best_acc))
                    print('Best loss: {:4f}'.format(best_loss))
                    model_name = MODEL_PREFIX + "_" + str(epoch) + "_" + str(time.time()) + ".pt"
                    torch.save(model_ft.state_dict(), os.path.join("checkpoints", model_name))
                    print("Saved model :", model_name)
                else:
                    patience += 1
                    print("Loss did not improve, patience: " + str(patience))

            # Early stopping
            if patience > LEARNING_PATIENCE:
                early_stop = True
                break

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val accuracy: {:4f}'.format(best_acc))

    training_history = {}
    training_history['train_acc'] = train_acc
    training_history['train_loss'] = train_loss
    training_history['val_acc'] = val_acc
    training_history['val_loss'] = val_loss

    return best_model, training_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

model_ft = models.resnet50(pretrained=True)
set_parameter_requires_grad(model_ft, True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
print(model_ft)

criterion = nn.CrossEntropyLoss()

if GPU_MODE:
    criterion.cuda()
    model_ft.cuda()

optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=BASE_LR)

# Run the functions and save the best model in the function model_ft.
model_ft, training_history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)

print("Training done")

try:
    with open(MODEL_PREFIX + '_training_history.pkl', 'wb') as handle:
        pickle.dump(training_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print("Exception in saving training history :" + str(e))

# Save model
model_name = MODEL_PREFIX + "_final_" + str(time.time()) + ".pt"
torch.save(model_ft.state_dict(), os.path.join("checkpoints", model_name))
print("Saved model :", model_name)