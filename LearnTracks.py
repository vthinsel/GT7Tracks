import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
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
args = parser.parse_args()

# path = r'C:\Users\Vince\OneDrive\Loisirs\SimRacing\SimTools\GT7\GT7Tracks\dumps'

all_files = glob.glob(os.path.join(args.folder, "*.csv"))
li = []
plt.xticks([])
plt.yticks([])
plt.title('Tracks')
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
    #plt.scatter(df['x'], df['z'], s=2)
    #plt.legend([os.path.basename(filename)])
# plt.show()
frame = pd.concat(li, axis=0, ignore_index=True)
frame.drop({'speed', 'rpm'}, axis=1, inplace=True)
frame.to_csv('tracks_concatenated.csv', index=False, decimal=".", sep=",")
# tracks_data = pd.get_dummies(frame, columns=["track_id"], drop_first=False)
# print(f"tracks_data columns: {tracks_data.columns}")
# print(f"tracks_data shape: {tracks_data.shape}")
# tracks_data.to_csv('tracks_cleaned.csv', index=False, decimal=".", sep=",")
# print(f"tracks_data shape: {tracks_data.shape}")

trackid_to_names = {119: 'track 1',
                346: 'track 2',
                363: 'track 3',
                4: 'track 4',
                837: 'track 5', }

print(f"trackidnames: {trackid_to_names}")
labels = frame['track_id']
inputs = frame.drop("track_id", axis=1)
#Normalize data
scaler = RobustScaler()
#scaler = normalize()
# fit and transform the data. We need to mke a dataframe again after the sklearn normalization which returns a numpy array
inputs = pd.DataFrame(scaler.fit_transform(inputs), columns=inputs.columns)
#Use scaler.scale_ to transform input for prediction ?

# Replace output values with range 0..number of tracks
val = 0
tracks_id = labels.unique()
track_dict = {}
for track_id in tracks_id:
    labels.replace(track_id, val, inplace=True)
    track_dict.update({val: trackid_to_names[track_id]})
    val += 1
tracks_id = labels.unique()
print(f"track_dict: {track_dict}\n")
print(f"inputs shape: {inputs.shape} labels shape: {labels.shape}")
x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, stratify=labels)
print(f"frame head:\n{frame.head()}\nx_train head:\n{x_train.head()}y_train head:\n{y_train.head()}")
print(f"frame shape: {frame.shape} x_train shape: {x_train.shape} y_train shape: {y_train.shape} unique_tracks: {len(tracks_id)}")

# Plot to check after normalization
plt.scatter(inputs['x'], inputs['z'], s=2)
plt.show()

model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(units=3, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=len(tracks_id), activation=tf.nn.softmax)
])
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.losses.sparse_categorical_crossentropy,
    metrics=["accuracy"]
)
print(model.summary())
model.fit(x_train, y_train, epochs=10)

# Evaluate model
print("Model Accuracy")
print(model.evaluate(x_test, y_test, return_dict=True))
model.save(filepath="gt7.model", overwrite=True)
