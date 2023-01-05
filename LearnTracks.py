import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
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

if args.refreshcsv:
    trackdefinitionurl = "https://raw.githubusercontent.com/ddm999/gt7info/web-new/_data/db/course.csv"
    print(f"Updating csv file from {trackdefinitionurl} for tracks definition")
    tracksdef = pd.read_csv(trackdefinitionurl)
    tracksdef.to_csv("track_list.csv", index=False, decimal=".", sep=",")
else:
    print("Using local CSV file for tracks definition")
    tracksdef = pd.read_csv("track_list.csv")
    # track name of track ID 1240: trackdef[trackdef["ID"] == 1240]['Name'].values[0]
print(f"Tracks definition:\n{tracksdef.head()}\n Tracks shape: {tracksdef.shape}")
all_files = glob.glob(os.path.join(args.folder, "*.csv"))
tracks_array = []
plt.title('Tracks from each files - 1 color per file')
plt.axis('equal')
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    trackid = df['track_id'].unique()[0]
    if trackid not in tracksdef.values:
        print(f"Reading file {filename} with track ID {trackid} is not part of track definition file")
    else:
        trackname = tracksdef[tracksdef["ID"] == trackid]['Name'].values[0]
        print(
            f"Reading file {filename} which has been recorded with ID {trackid} ({trackname}) with {len(df)} data coordinates")
        tracks_array.append(df)
        plt.scatter(df['x'], df['z'], s=2, label=trackname)
plt.legend(loc='upper left')
plt.show()
# fig, axes = plt.subplots(len(tracks_array),figsize=(8, 8))
# fig.suptitle('Vertically stacked subplots')
# graph = 0
# for track in tracks_array:
#    axes[graph].scatter(track['x'], track['z'], s=1)
#    axes[graph].set_title(graph)
#    graph += 1
# plt.show()
frame = pd.concat(tracks_array, axis=0, ignore_index=True)
frame.drop({'speed', 'rpm'}, axis=1, inplace=True)
frame.to_csv('tracks_concatenated.csv', index=False, decimal=".", sep=",")
# tracks_data = pd.get_dummies(frame, columns=["track_id"], drop_first=False)
# print(f"tracks_data columns: {tracks_data.columns}")
# print(f"tracks_data shape: {tracks_data.shape}")
# tracks_data.to_csv('tracks_cleaned.csv', index=False, decimal=".", sep=",")
# print(f"tracks_data shape: {tracks_data.shape}")

y = frame['track_id']
x = frame.drop("track_id", axis=1)
# Normalize data
scaler = RobustScaler()
# fit and transform the data. We need to make a dataframe again after the sklearn normalization which returns a numpy array
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
print(f"Scaler factors: {scaler.scale_}")
# Use scaler.scale_ to transform input for prediction ?

# Replace output values with range 0..number of tracks
val = 0
tracks_id = y.unique()
track_dict = {}
for track_id in tracks_id:
    y.replace(track_id, val, inplace=True)
    track_dict.update({val: tracksdef[tracksdef["ID"] == track_id]['Name'].values[0]})
    val += 1
print(f"track_dict: {track_dict}\n")
print(f"inputs shape: {x.shape} labels shape: {y.shape}")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y)
print(f"frame head:\n{frame.head()}\nx_train head:\n{x_train.head()}\ny_train head:\n{y_train.head()}")
print(
    f"frame shape: {frame.shape} x_train shape: {x_train.shape} y_train shape: {y_train.shape} unique_tracks: {len(tracks_id)}")

# Plot to check after normalization
plt.title('Tracks merged and normalized')
plt.scatter(x['x'], x['z'], s=2)
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
# model.save(filepath="gt7.model", overwrite=True)
sample_multipletracks = pd.DataFrame(np.array(
    [
        [0.0, 0.0, 0.0],
        [-0.013608, 0.042855, 0.080122],
        [-0.137124, -0.424976, -0.622008],
        [0.1984, -0.12, 0],
        [1.3, -0.65, -1.231429],  # between 3 tracks
        [-1.0, 0.7, 0.5],
        [0.1636, 0.0606, 0.5],
        [2.5, 1.0, 0]  # dragon tail
    ]),
    columns=['x', 'z', 'y']
)
res = model.predict(sample_multipletracks)
print(f"Points from various tracks:\n{res}")
for row in res:
    print(f"Track index is {np.argmax(row)} which is {row[np.argmax(row)] * 100}% {track_dict[np.argmax(row)]}")
sample_singletracks = pd.DataFrame(np.array(
    [
        [-0.250120, -0.230890, -0.220905],  # 50/50
        [0.688583, -0.739107, -0.781785],  # GP
        [-0.398436, -0.055111, 0.310404],  # GP
        [0.556314, -0.541925, -1.231429],  # 50/50
    ]),
    columns=['x', 'z', 'y']
)
res = model.predict(sample_singletracks)
print(
    f"Points belonging to the same track:\n{res}")
for row in res:
    print(f"Track index is {np.argmax(row)} which is {row[np.argmax(row)] * 100}% {track_dict[np.argmax(row)]}")
