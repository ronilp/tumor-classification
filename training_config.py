import torch

### Learning Parameters
BASE_LR = 1e-4
LR_DECAY_EPOCHS = 10
LR_DECAY = 0.1

### Dataset Config
NUM_CLASSES = 1
# DATA_DIR = '/Users/rpancholia/Documents/Acads/Projects/data/pbt-classification/'
DATA_DIR = '/Users/rpancholia/Documents/Acads/Projects/data/data_T2_only_2'
AGE_CSV_PATH = '/Users/rpancholia/Documents/Acads/Projects/data/flipped_clinical_NormalPedBrainAge_StanfordCohort.csv'
ALLOWED_CLASSES = ["Normals"]
MODEL_FILTER = "T2 ax"
SINGLE_CHANNEL = False

### Miscellaneous Config
MODEL_PREFIX = "age_regression"
BATCH_SIZE = 15

### GPU SETTINGS
CUDA_DEVICE = 0  # GPU device ID
GPU_MODE = torch.cuda.is_available()
