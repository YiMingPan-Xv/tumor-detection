"""
This script downloads the dataset from Kaggle, processes the images, and saves them in a CSV file.

The dataset is a collection of brain MRI images for brain tumor detection.
The images are divided into two categories: "yes" (tumor present) and "no" (tumor absent).
The script performs the following steps:
1. Downloads the dataset from Kaggle using the `kagglehub` library.
2. Reads the images from the "yes" and "no" folders.
3. Converts the images to arrays and resizes them to a common size (128x128).
4. Converts the images to grayscale.
5. Creates a DataFrame with the images and their corresponding labels (1 for "yes" and 0 for "no").
6. Shuffles the DataFrame.
7. Saves the DataFrame as a CSV file named "tumor_data.csv".
"""

import glob
import os
from pathlib import Path

import kagglehub
import pandas as pd


def get_data():

    # Download latest version
    path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")

    yes_path = path + "/yes"
    no_path = path + "/no"

    # Convert images to array
    yes_images = glob.glob(yes_path + "/*.jpg")
    no_images = glob.glob(no_path + "/*.jpg")

    return yes_images, no_images

def make_df(images, labels):
    data = []
    for i in range(len(images)):
        data.append([images[i], labels[i]])
    df = pd.DataFrame(data, columns=["path", "label"])
    return df

def initialize(force=False):
    # Set the working directory to the project directory
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent # assuming src is one level deep
    os.chdir(project_dir)

    # Lazy load the dataset if it exists
    if os.path.exists("data/tumor_data.csv") and not force:
        print("Using existing dataset")
        return pd.read_csv("data/tumor_data.csv")

    yes_images, no_images = get_data()

    yes_labels = [1] * len(yes_images)
    no_labels = [0] * len(no_images)

    df_yes = make_df(yes_images, yes_labels)
    df_no = make_df(no_images, no_labels)

    df = pd.concat([df_yes, df_no], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataframe

    df.to_csv("data/tumor_data.csv", index=False)
    print("Dataframe saved as tumor_data.csv")

    return df
