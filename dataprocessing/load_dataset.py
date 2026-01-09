"""
load_dataset.py — Dataset preparation for Traffic Sign Recognition (GTSRB)
Author: G Sreekanth
Description:
    - Downloads GTSRB dataset from Kaggle
    - Extracts images into 'train' and 'test' directories    
"""

import os
import zipfile
import tensorflow as tf

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path("dataset/raw/gtsrb")
TRAIN_DIR = DATA_DIR / "Train"
TEST_DIR = DATA_DIR / "Test"
IMG_SIZE = (30, 30)
BATCH_SIZE = 32
VALID_SPLIT = 0.3


# ============================================================
# 1. Download dataset (optional if not present)
# ============================================================
def download_gtsrb_from_kaggle():
    """
    Downloads the GTSRB dataset from Kaggle if kaggle.json is configured.
    Note: Make sure kaggle.json is placed in ~/.kaggle/ before running.
    """
    try:
        print("🔽 Downloading GTSRB dataset from Kaggle...")
        api = KaggleApi()
        api.authenticate()

        dataset = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"        
        api.dataset_download_files(dataset, path=DATA_DIR, unzip=True)
        #with zipfile.ZipFile("gtsrb-german-traffic-sign.zip", "r") as zip_ref:
        #    zip_ref.extractall("data")
        print("✅ Dataset downloaded and extracted successfully! ",os.path.abspath(DATA_DIR))
    except Exception as e:
        print("⚠️ Could not download from Kaggle. Please place the dataset manually under /data.")
        print(f"Error: {e}")



# ============================================================
# 
# ============================================================
if __name__ == "__main__":
    print("Checking for GTSRB dataset...")
    if not TRAIN_DIR.exists():
        print("Training directory not found. Downloading dataset...")
        download_gtsrb_from_kaggle()
    else:
        print(os.path.abspath(DATA_DIR))
        print("Training directory found. Skipping download.",TRAIN_DIR)

