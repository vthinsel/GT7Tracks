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

path = r'C:\Users\Vince\OneDrive\Loisirs\SimRacing\SimTools\GT7\GT7Tracks\dumps'
all_files = glob.glob(os.path.join(path, "*.csv"))
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
frame = pd.concat(li, axis=0, ignore_index=True)
frame.drop({'speed', 'rpm'}, axis=1, inplace=True)
frame.to_csv('data_tracks.csv')
tracks_data = pd.get_dummies(frame, columns=["track_id"], drop_first=False)
print(f"tracks_data columns: {tracks_data.columns}")
print(f"tracks_data shape: {tracks_data.shape}")
print(tracks_data.shape)
print(tracks_data.head())

