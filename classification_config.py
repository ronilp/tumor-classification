import torch

### Learning Parameters
BASE_LR = 0.001
LR_DECAY_EPOCHS = 10
LR_DECAY = 0.1

### Dataset Config
NUM_CLASSES = 2
DATA_DIR = '/Users/rpancholia/Documents/Acads/Projects/data/pbt-classification/'
ALLOWED_CLASSES = ["DIPG", "Normals"]
MODEL_FILTER = "T2 ax"
SINGLE_CHANNEL = False

### Miscellaneous Config
BATCH_SIZE = 10

### GPU SETTINGS
CUDA_DEVICE = 0  # GPU device ID
GPU_MODE = torch.cuda.is_available()
