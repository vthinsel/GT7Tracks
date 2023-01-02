import argparse

import pandas as pd
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

parser = argparse.ArgumentParser()
parser.add_argument("--folder",
                    required=True,
                    type=str,
                    default=".",
                    help="Folder containing csv files to use for learning")

parser.add_argument("--outfolder",
                    required=False,
                    type=str,
                    default=".",
                    help="Folder containing outputfiles")
args = parser.parse_args()

#path = r'C:\Users\Vince\OneDrive\Loisirs\SimRacing\SimTools\GT7\GT7Tracks\dumps'

all_files = glob.glob(os.path.join(args.folder, "*.csv"))
li = []
plt.xticks([])
plt.yticks([])

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
    plt.scatter(df['x'], df['z'], s=2)
plt.legend()
plt.show()
frame = pd.concat(li, axis=0, ignore_index=True)
frame.drop({'speed', 'rpm'}, axis=1, inplace=True)
frame.to_csv('tracks_concatenated.csv', index=False, decimal=".", sep=",")
tracks_data = pd.get_dummies(frame, columns=["track_id"], drop_first=False)
print(f"tracks_data columns: {tracks_data.columns}")
print(f"tracks_data shape: {tracks_data.shape}")
print(tracks_data.head())
tracks_data.to_csv('tracks_cleaned.csv', index=False, decimal=".", sep=",")
