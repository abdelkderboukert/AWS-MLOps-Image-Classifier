import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ZIP_PATH = os.path.join(DATA_DIR, 'raw', 'archive.zip')
EXTRACT_PATH = os.path.join(DATA_DIR, 'processed')
CSV_PATH = os.path.join(EXTRACT_PATH, 'train.csv')

# Image Parameters
IMG_SIZE = (224, 224)
N_FEATURES_ORB = 500

# BoVW Parameters
VOCAB_SIZE = 500  # K for KMeans
PCA_COMPONENTS = 100

# Model Parameters
RANDOM_STATE = 42
CV_FOLDS = 8
TEST_SIZE = 0.2