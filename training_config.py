import torch
from utils.constants import GE, SCANNER_15T, SCANNER_3T, SIEMENS, MEAN, STD_DEV, PHILIPS, MAX, CROP_BOTTOM, CROP_CENTER

### Learning Parameters
BASE_LR = 1e-4
LR_DECAY_EPOCHS = 10
LR_DECAY = 0.1
LEARNING_PATIENCE = 10
WEIGHTED_LOSS_ON = True

### Dataset Config
DATA_DIR = '/Users/rpancholia/Documents/Acads/Projects/data/classification-pkl-augmented/'
INPUT_DCM_PATH = "../data/classification-dcm/"
OUTPUT_PREPROCESS_PATH = "../data/classification-pkl-augmented/"
ALLOWED_CLASSES = ["MB", "DIPG", "EP"]
NUM_CLASSES = len(ALLOWED_CLASSES)
# set as False to convert 2D image to 3D RGB image
SINGLE_CHANNEL = False
CLASS_LIMIT = 500000

# regression configs
MODEL_FILTER = "T2 ax"
AGE_CSV_PATH = '/Users/rpancholia/Documents/Acads/Projects/data/flipped_clinical_NormalPedBrainAge_StanfordCohort.csv'

### Preprocessing Config
CROP_SIZE = 224
# set crop mid point as either CROP_CENTER or CROP_BOTTOM
CROP_MID_POINT = CROP_BOTTOM
INTENSITY_Z_SCORE_NORM = False
INTENSITY_01_NORM = True

### Miscellaneous Config
MODEL_PREFIX = "dipg_vs_mb_vs_eb"
BATCH_SIZE = 1
EARLY_STOPPING_ENABLED = True
MAX_PIXEL_VAL = 255
RANDOM_SEED = 150
SAVE_EVERY_MODEL = False

### GPU SETTINGS
CUDA_DEVICE = 0  # GPU device ID
GPU_MODE = torch.cuda.is_available()

### Normalization Values
INTENSITY_DICT = {
    GE: {
        SCANNER_15T: {
            MEAN: 1742.24,
            STD_DEV: 900.807,
            MAX: 5537.0
        },
        SCANNER_3T: {
            MEAN: 5157.49,
            STD_DEV: 1518.842,
            MAX: 10599.0
        }
    },
    PHILIPS: {
        SCANNER_15T: {
            MEAN: 731.01,
            STD_DEV: 502.012,
            MAX: 2807.0
        },
        SCANNER_3T: {
            MEAN: 479.00,
            STD_DEV: 26.092,
            MAX: 525.0
        }
    },
    SIEMENS: {
        SCANNER_15T: {
            MEAN: 1549.08,
            STD_DEV: 657.38,
            MAX: 3615.0
        },
        SCANNER_3T: {
            MEAN: 2374.44,
            STD_DEV: 291.936,
            MAX: 3062.0
        }
    }
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STDDEV = [0.229, 0.224, 0.225]