import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

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
parser.add_argument("--refreshcsv",
                    required=False,
                    type=str,
                    default=False,
                    help="Refresh track csv from github")
args = parser.parse_args()

# path = r'C:\Users\Vince\OneDrive\Loisirs\SimRacing\SimTools\GT7\GT7Tracks\dumps'

if args.refreshcsv == True:
    print(f"Updating csv file from {trackdefinitionurl} for tracks definition")
    trackdefinitionurl = "https://raw.githubusercontent.com/ddm999/gt7info/web-new/_data/db/course.csv"
    tracksdef = pd.read_csv(trackdefinitionurl)
    tracksdef.to_csv("track_list.csv", index=False, decimal=".", sep=",")
else:
    print("Using local CSV file for tracks definition")
    tracksdef = pd.read_csv("track_list.csv")
    #track name of track ID 1240: trackdef[trackdef["ID"] == 1240]['Name']
print(f"Tracks definition:\n{tracksdef.head()}")
all_files = glob.glob(os.path.join(args.folder, "*.csv"))
tracks_array = []
plt.title('Tracks from each files - 1 color per file')
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    trackid = df['track_id'].unique()[0]
    trackname = tracksdef[tracksdef["ID"] == trackid]['Name'].values[0]
    print(f"Reading file {filename} which has been recorded with ID {trackid} ({trackname}) with {len(df)} data coordinates")
    tracks_array.append(df)
    plt.scatter(df['x'], df['z'], s=2)
plt.show()
frame = pd.concat(tracks_array, axis=0, ignore_index=True)
frame.drop({'speed', 'rpm'}, axis=1, inplace=True)
frame.to_csv('tracks_concatenated.csv', index=False, decimal=".", sep=",")
# tracks_data = pd.get_dummies(frame, columns=["track_id"], drop_first=False)
# print(f"tracks_data columns: {tracks_data.columns}")
# print(f"tracks_data shape: {tracks_data.shape}")
# tracks_data.to_csv('tracks_cleaned.csv', index=False, decimal=".", sep=",")
# print(f"tracks_data shape: {tracks_data.shape}")

labels = frame['track_id']
inputs = frame.drop("track_id", axis=1)
# Normalize data
scaler = RobustScaler()
# scaler = normalize()
# fit and transform the data. We need to mke a dataframe again after the sklearn normalization which returns a numpy array
inputs = pd.DataFrame(scaler.fit_transform(inputs), columns=inputs.columns)
# Use scaler.scale_ to transform input for prediction ?

# Replace output values with range 0..number of tracks
val = 0
tracks_id = labels.unique()
track_dict = {}
for track_id in tracks_id:
    labels.replace(track_id, val, inplace=True)
    track_dict.update({val: tracksdef[tracksdef["ID"] == track_id]['Name'].values[0]})
    val += 1
print(f"track_dict: {track_dict}\n")
print(f"inputs shape: {inputs.shape} labels shape: {labels.shape}")
x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, stratify=labels)
print(f"frame head:\n{frame.head()}\nx_train head:\n{x_train.head()}\ny_train head:\n{y_train.head()}")
print(
    f"frame shape: {frame.shape} x_train shape: {x_train.shape} y_train shape: {y_train.shape} unique_tracks: {len(tracks_id)}")

# Plot to check after normalization
plt.title('Tracks merged and normalized')
plt.scatter(inputs['x'], inputs['z'], s=2)
plt.show()

model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(units=3, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=len(tracks_id), activation=tf.nn.softmax)
])
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.losses.sparse_categorical_crossentropy,
    metrics=["accuracy"]
)
print(model.summary())
model.fit(x_train, y_train, epochs=30)

# Evaluate model
print("Model Accuracy")
print(model.evaluate(x_test, y_test, return_dict=True))
model.save(filepath="gt7.model", overwrite=True)
