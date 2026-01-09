
import os

import cv2
import keras
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

"""
preprocess.py — Data Preprocessing for Traffic Sign Recognition (GTSRB)
Author: G Sreekanth 
Description:
    - Preprocesses GTSRB dataset by resizing images, splitting into training and validation sets, converts class vector to one-hot encoded and saving as numpy arrays.

"""
#===========================================================
def preprocess_data(data_dir='dataset/raw/gtsrb',root_dir='dataset', IMG_HEIGHT=30, IMG_WIDTH=30):
    image_data = []
    image_labels = []

    NUM_CATEGORIES = len(os.listdir(data_dir + '/Train'))
    for i in range(NUM_CATEGORIES):
        path = data_dir + '/Train/' + str(i)
        images = os.listdir(path)

        for img in images:
            try:
                image = cv2.imread(path + '/' + img)
                image_fromarray = Image.fromarray(image, 'RGB')
                resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
                image_data.append(np.array(resize_image))
                image_labels.append(i)
            except:
                print("Error in " + img)

    # Changing the list to numpy array
    image_data = np.array(image_data)
    image_labels = np.array(image_labels)

    print(image_data.shape, image_labels.shape)
    # Splitting training and testing dataset
    X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.3, random_state=42,shuffle=True)
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # One hot encoding for labels
    y_train = keras.utils.to_categorical(y_train, NUM_CATEGORIES)
    y_val = keras.utils.to_categorical(y_val, NUM_CATEGORIES)


    # Save preprocessed data
    preprocess_data_dir = root_dir + '/preprocessed'
    os.makedirs(preprocess_data_dir, exist_ok=True)
    np.save(os.path.join(preprocess_data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(preprocess_data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(preprocess_data_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(preprocess_data_dir, 'y_val.npy'), y_val)

    print("----------- Data Preprocessing Complete -----------")


def prepare_test_data(data_dir='dataset/raw/gtsrb',root_dir='dataset', IMG_HEIGHT=30, IMG_WIDTH=30):
    test_image_data = []
    test_image_labels = []
    
    test = pd.read_csv(data_dir + '/Test.csv')
    imgs = test["Path"].values
    labels = test["ClassId"].values

    for i in range(len(imgs)):
        try:
            image = cv2.imread(data_dir + '/' + imgs[i])
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
            test_image_data.append(np.array(resize_image))
            test_image_labels.append(labels[i])
        except:
            print("Error in " + imgs[i])
    
    test_image_data = np.array(test_image_data)
    test_image_labels = np.array(test_image_labels)
    test_image_data = test_image_data/255
    test_image_labels = keras.utils.to_categorical(test_image_labels, 43)

    preprocess_data_dir = root_dir + '/preprocessed'
    os.makedirs(preprocess_data_dir, exist_ok=True)    
    np.save(os.path.join(preprocess_data_dir, 'X_test.npy'), test_image_data)
    np.save(os.path.join(preprocess_data_dir, 'y_test.npy'), test_image_labels)
    print("----------- Test Data Preprocessing Complete -----------")

if __name__ == "__main__":
    preprocess_data()
    prepare_test_data()