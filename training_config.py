import torch
from utils.constants import *

### Learning Parameters
BASE_LR = 1e-4
LR_DECAY_EPOCHS = 10
LR_DECAY = 0.1
LEARNING_PATIENCE = 10
WEIGHTED_LOSS_ON = True

### Dataset Config
NUM_CLASSES = 3
DATA_DIR = '/Users/rpancholia/Documents/Acads/Projects/data/classification-pkl-augmented/'
AGE_CSV_PATH = '/Users/rpancholia/Documents/Acads/Projects/data/flipped_clinical_NormalPedBrainAge_StanfordCohort.csv'
ALLOWED_CLASSES = ["MB", "DIPG", "EP"]
MODEL_FILTER = "T2 ax"
SINGLE_CHANNEL = False
CLASS_LIMIT = 5000

### Miscellaneous Config
MODEL_PREFIX = "dipg_vs_mb_vs_eb"
BATCH_SIZE = 1
EARLY_STOPPING_ENABLED = False
MAX_PIXEL_VAL = 255
RANDOM_SEED = 150

### GPU SETTINGS
CUDA_DEVICE = 0  # GPU device ID
GPU_MODE = torch.cuda.is_available()

### Normalization Values
INTENSITY_DICT = {
    GE: {
        SCANNER_15T: {
            MEAN: 1742.24,
            STD_DEV: 900.807
        },
        SCANNER_3T: {
            MEAN: 5157.49,
            STD_DEV: 1518.842
        }
    },
    PHILIPS: {
        SCANNER_15T: {
            MEAN: 731.01,
            STD_DEV: 502.012
        },
        SCANNER_3T: {
            MEAN: 479.00,
            STD_DEV: 26.092
        }
    },
    SIEMENS: {
        SCANNER_15T: {
            MEAN: 1549.08,
            STD_DEV: 657.38
        },
        SCANNER_3T: {
            MEAN: 2374.44,
            STD_DEV: 291.936
        }
    }
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STDDEV = [0.229, 0.224, 0.225]