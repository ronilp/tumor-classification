import torch

### Learning Parameters
BASE_LR = 1e-4
LR_DECAY_EPOCHS = 10
LR_DECAY = 0.1
LEARNING_PATIENCE = 10

### Dataset Config
NUM_CLASSES = 2
DATA_DIR = '/Users/rpancholia/Documents/Acads/Projects/data/pbt-classification/'
AGE_CSV_PATH = '/Users/rpancholia/Documents/Acads/Projects/data/flipped_clinical_NormalPedBrainAge_StanfordCohort.csv'
ALLOWED_CLASSES = ["MB", "DIPG"]
MODEL_FILTER = "T2 ax"
SINGLE_CHANNEL = False
CLASS_LIMIT = 1000

### Miscellaneous Config
MODEL_PREFIX = "dipg_vs_mb"
BATCH_SIZE = 1
EARLY_STOPPING_ENABLED = False
MAX_PIXEL_VAL = 255
RANDOM_SEED = 150

### GPU SETTINGS
CUDA_DEVICE = 0  # GPU device ID
GPU_MODE = torch.cuda.is_available()
