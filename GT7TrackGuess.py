import argparse
import json
import socket

import pandas as pd
import tensorflow as tf
from salsa20 import Salsa20_xor

from gt_packet_definition import GTDataPacket


# import os
# import logging
# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# os.environ["KMP_AFFINITY"] = "noverbose"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_trackdict(filename):
    with open(filename) as f:
        trackdict = json.loads(f.read())
    return trackdict


def salsa20_dec(dat):
    KEY = b'Simulator Interface Packet GT7 ver 0.0'
    # Seed IV is always located here
    oiv = dat[0x40:0x44]
    iv1 = int.from_bytes(oiv, byteorder='little')
    # Notice DEADBEAF, not DEADBEEF
    iv2 = iv1 ^ 0xDEADBEAF
    IV = bytearray()
    IV.extend(iv2.to_bytes(4, 'little'))
    IV.extend(iv1.to_bytes(4, 'little'))
    ddata = Salsa20_xor(dat, bytes(IV), KEY[0:32])
    magic = int.from_bytes(ddata[0:4], byteorder='little')
    if magic != 0x47375330:
        return bytearray(b'')
    return ddata


# send heartbeat

def send_hb(s):
    send_data = 'A'
    s.sendto(send_data.encode('utf-8'), (args.ps_ip, SendPort))


parser = argparse.ArgumentParser()
parser.add_argument("--folder",
                    required=False,
                    type=str,
                    default="dumps",
                    help="Folder containing csv files to use for learning")
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
parser.add_argument("--ps_ip",
                    required=True,
                    type=str,
                    help="Playstation 4/5 IP address. Accepts IP or FQDN provided it resolves to something.")
args = parser.parse_args()

# ports for send and receive data
SendPort = 33739
ReceivePort = 33740
# Create a UDP socket and bind it to connect to GT7
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', ReceivePort))
s.settimeout(5)

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
model = tf.keras.models.load_model(args.modelname)
track_dict = load_trackdict('trackdict.json')
track_confidence = pd.DataFrame(columns=track_dict.values())
# plt.ion()  # allows us to continue to update the plot
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.axis('off')  # hides the black border around the axis.
# plt.xticks([])
# plt.yticks([])
# px, pz, py = None, None, None

# print(model.summary())
print(f"I can recognize the following tracks: {track_dict.values()}")
candidate_track = ""
track_score = 0.0
pktid = 0
pknt = 0
while True:
    try:
        data, address = s.recvfrom(4096)
        pknt = pknt + 1
        ddata = salsa20_dec(data)
        telemetry = GTDataPacket(ddata[0:296])
        if len(ddata) > 0 and telemetry.pkt_id > pktid and pknt > 100:
            pktid = telemetry.pkt_id
            send_hb(s)
            pknt = 0
            data = [[telemetry.position_x, telemetry.position_z, telemetry.position_y, telemetry.northorientation, telemetry.rotation_x, telemetry.rotation_z, telemetry.rotation_y]]
            #print(f"Coord: {coord}")
            # x, z, y = telemetry.position_x, telemetry.position_z, telemetry.position_y
            # if px is None: =
            #    px, pz, py = x, z, y
            #    continue
            # speed = min(1, telemetry.speed / telemetry.calculated_max_speed) * 3
            # color = plt.cm.plasma(speed)
            # plt.plot([px, x], [pz, z], color=color)
            # plt.gca().set_aspect('equal', adjustable='box')
            # plt.pause(0.00000000000000000001)
            #px, pz, py = x, z, y
            res = model.predict(data, verbose=0)
            for row in res:
                track_confidence.loc[len(track_confidence.index)] = row
                track_mean = track_confidence.mean(axis=0)
                updated_candidate_track = track_mean.idxmax()
                updated_track_score = track_mean.max()
                if candidate_track != updated_candidate_track:
                    candidate_track = updated_candidate_track
                    print(f"Tracks mean confidence: {track_mean*100:.2f}%")
                    print(f"Track candidate : {candidate_track} with confidence {(track_mean.max() * 100):.1f}% ")
                if updated_track_score > track_score:
                    track_score = updated_track_score
                    print(f"Track candidate : {candidate_track} with higher confidence {(track_mean.max() * 100):.1f}% ")

    except Exception as e:
        # print(e)
        send_hb(s)
        pknt = 0
        pass
