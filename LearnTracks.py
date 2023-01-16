import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def createmodel(num_outputs, num_inputs, units):
    newmodel = tf.keras.Sequential([
        tf.keras.Input(shape=(num_inputs,)),
        tf.keras.layers.Dense(units=num_inputs, activation=tf.nn.relu),
        # tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=units, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=num_outputs, activation=tf.nn.softmax)
    ])
    newmodel.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"]
    )
    return newmodel

def save_trackdict(trackdict, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(trackdict))


def load_trackdict(filename):
    with open(filename) as f:
        trackdict = json.loads(f.read())
    return trackdict

def save_scalefactor(factor, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(factor))


def load_scalefactor(filename):
    with open(filename) as f:
        factor = json.loads(f.read())
    return factor

parser = argparse.ArgumentParser()
parser.add_argument("--folder",
                    required=False,
                    type=str,
                    default="dumps",
                    help="Folder containing csv files to use for learning")
parser.add_argument("--mode",
                    required=False,
                    type=str,
                    default="predict",
                    help="train or predict mode")
parser.add_argument("--modelname",
                    required=False,
                    type=str,
                    default="gt7.model",
                    help="model name to load/save")
parser.add_argument("--refreshcsv",
                    required=False,
                    type=str,
                    default=False,
                    help="Refresh track csv from github")
args = parser.parse_args()

# path = r'C:\Users\Vince\OneDrive\Loisirs\SimRacing\SimTools\GT7\GT7Tracks\dumps'

if args.mode == "train":
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
    #frame.drop({'speed', 'rpm'}, axis=1, inplace=True)
    columns_to_drop = [
        "speed",
        "rpm",
        "track_id",
        #"rotation_x",
        #"rotation_z",
        #"rotation_y"
    ]
    x = frame.drop(columns=columns_to_drop, axis=1)
    frame.to_csv('tracks_concatenated.csv', index=False, decimal=".", sep=",")
    y = frame['track_id']
    # Normalize data
    #scaler = RobustScaler()
    #x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    #print(f"Scaler factors: {scaler.scale_}")

    # Replace output values with range 0..number of tracks
    val = 0
    tracks_id = y.unique()
    track_dict = {}
    for track_id in tracks_id:
        y.replace(track_id, val, inplace=True)
        track_dict.update({str(val): tracksdef[tracksdef["ID"] == track_id]['Name'].values[0]})
        val += 1
    print(f"track_dict: {track_dict}\n")
    print(f"inputs shape: {x.shape} labels shape: {y.shape}")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y)
    print(f"frame head:\n{frame.head()}\nx_train head:\n{x_train.head()}\ny_train head:\n{y_train.head()}")
    print(
        f"frame shape: {frame.shape} x_train shape: {x_train.shape} y_train shape: {y_train.shape} unique_tracks: {len(tracks_id)}")
    # Plot to check after normalization
    plt.title('Tracks merged and normalized')
    plt.axis('equal')
    plt.scatter(x['x'], x['z'], s=2)
    plt.show()
    model = createmodel(len(tracks_id), 7, 256)
    model.fit(x_train, y_train, epochs=50)
    # Evaluate model
    print("Model Accuracy")
    print(model.evaluate(x_test, y_test, return_dict=True))
    model.save(filepath=args.modelname, overwrite=True)
    save_trackdict(track_dict, 'trackdict.json')
else:
    model = tf.keras.models.load_model(args.modelname)
    track_dict = load_trackdict('trackdict.json')

#print(model.summary())
#print(track_dict)

#livetrack = pd.read_csv("dumps\\119.csv", index_col=None, header=0) #GP
#livetrack = pd.read_csv("dumps\\346.csv", index_col=None, header=0) #Indy
#livetrack = pd.read_csv("dumps\\4.csv", index_col=None, header=0) #daytona
#livetrack = pd.read_csv("dumps\\363.csv", index_col=None, header=0) #dragon tail
#livetrack = pd.read_csv("dumps\\837.csv", index_col=None, header=0) # Fuji Intl

livetrack = pd.read_csv("dumps\\1232.csv", index_col=None, header=0) #Autodrome Lago Maggiore - West

livetrack.drop({'speed', 'rpm', 'track_id'}, axis=1, inplace=True)
#livetrack.div({'x':425.1415081 , 'z':543.85843277 , 'y':12.19364524})
print(livetrack.shape)
print(livetrack.head())
track_confidence = pd.DataFrame(columns=track_dict.values())
candidate_track = ""
res = model.predict(livetrack)
for row in res:
    track_confidence.loc[len(track_confidence.index)] = row
    track_mean = track_confidence.mean(axis=0)
    updated_candidate_track = track_mean.idxmax()
    if candidate_track != updated_candidate_track:
        candidate_track = updated_candidate_track
        print(f"Track candidate : {candidate_track} with confidence {(track_mean.max()*100):.1f}% ")
    #print(f"Point Track index is {np.argmax(row)} which is {(row[np.argmax(row)]*100):.1f}% {track_dict[str(np.argmax(row))]}")
pd.options.display.float_format = '{:,.2f}'.format
print(f"\nFinal Mean:\n {track_confidence.mean(axis=0)}")
#print(f"\nMax:\n {track_confidence.max(axis=0)}")
#print(f"\nMin:\n {track_confidence.min(axis=0)}")

