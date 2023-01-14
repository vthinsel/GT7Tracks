import argparse
import csv
import datetime
import pickle
import signal
import socket
import sys

import matplotlib.pyplot as plt
from salsa20 import Salsa20_xor

from gt_packet_definition import GTDataPacket

# ansi prefix
pref = "\033["


def handler(signum, frame):
    sys.stdout.write(f'{pref}?1049l')  # revert buffer
    sys.stdout.write(f'{pref}?25h')  # restore cursor
    sys.stdout.flush()
    exit(1)


# handle ctrl-c
signal.signal(signal.SIGINT, handler)

sys.stdout.write(f'{pref}?1049h')  # alt buffer
sys.stdout.write(f'{pref}?25l')  # hide cursor
sys.stdout.flush()

# ports for send and receive data
SendPort = 33739
ReceivePort = 33740

# ctrl-c handler

parser = argparse.ArgumentParser()
parser.add_argument("--ps_ip",
                    required=True,
                    type=str,
                    help="Playstation 4/5 IP address. Accepts IP or FQDN provided it resolves to something.")

parser.add_argument("--logpackets",
                    type=bool,
                    default=False,
                    help="Optionnaly log packets for future playback using https://github.com/vthinsel/Python_UDP_Receiver/UDPSend_timed.py .Default is False")

parser.add_argument("--track",
                    required=True,
                    type=str,
                    default="",
                    help="Track ID as per https://github.com/ddm999/gt7info/blob/web-new/_data/db/course.csv")

args = parser.parse_args()

# Create a UDP socket and bind it to connect to GT7
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', ReceivePort))
s.settimeout(5)


# data stream decoding


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


# generic print function
def printAt(str, row=1, column=1, bold=0, underline=0, reverse=0):
    sys.stdout.write('{}{};{}H'.format(pref, row, column))
    if reverse:
        sys.stdout.write('{}7m'.format(pref))
    if bold:
        sys.stdout.write('{}1m'.format(pref))
    if underline:
        sys.stdout.write('{}4m'.format(pref))
    if not bold and not underline and not reverse:
        sys.stdout.write('{}0m'.format(pref))
    sys.stdout.write(str)


# start by sending heartbeat to wake-up GT7 telemetry stack
send_hb(s)
# setup the plot styling
plt.ion()  # allows us to continue to update the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.axis('off')  # hides the black border around the axis.
plt.xticks([])
plt.yticks([])
px, pz, py = None, None, None

printAt('GT7 Track Recorder 1.0 (ctrl-c to quit)', 1, 1, bold=1)
printAt('Packet ID:', 3, 40)
printAt('{:<60}'.format('Current Car Data'), 3, 1, reverse=1, bold=1)
printAt('Car ID:', 3, 28, reverse=1)
printAt('Pkt ID:', 3, 43, reverse=1)  # packet id
printAt('X:', 4, 1)
printAt('Y:', 4, 11)
printAt('Z:', 4, 21)

printAt('Throttle:    %', 5, 1)
printAt('RPM:        ', 5, 21)
printAt('Speed:        km/h', 5, 41)
sys.stdout.flush()

# Create output files if needed
if args.logpackets:
    f1 = open(args.track + '.cap', 'wb')
    f2 = open(args.track + '.raw', 'wb')
csvfile = open("dumps/new/" +args.track + '.csv', 'w', newline='')
csvwriter = csv.writer(csvfile)

pktid = 0
pknt = 0
previousts = datetime.datetime.now()
delta = 0
csvheader = True
slip_angle = 0

while True:
    try:
        data, address = s.recvfrom(4096)
        if pknt == 0:  # Init time reference upon first packet received
            previousts = datetime.datetime.now()
        pknt += 1
        ts = datetime.datetime.now()
        delta = ts - previousts
        previousts = ts
        if args.logpackets:
            previoustime = ('{:%H:%M:%S:%f}'.format(datetime.datetime.now()))
            record = [previoustime, delta, data]
            pickle.dump(record, f1)
            f2.write(data)
        ddata = salsa20_dec(data)
        telemetry = GTDataPacket(ddata[0:296])
        if len(ddata) > 0 and telemetry.pkt_id > pktid:
            x, z, y = telemetry.position_x, telemetry.position_z, telemetry.position_y
            if px is None:
                px, pz, py = x, z, y
                continue
            # here we're getting the ratio of how fast the car's going compared to it's max speed.
            # we're multiplying by 3 to boost the colorization range.
            speed = min(1, telemetry.speed / telemetry.calculated_max_speed) * 3
            # Now use the "speed" ratio to select the color from the Matplotlib pallet
            color = plt.cm.plasma(speed)
            # plot the current step
            plt.plot([px, x], [pz, z], color=color)
            # set the aspect ratios to be equal for x/z axis, this way the map doesn't look skewed
            plt.gca().set_aspect('equal', adjustable='box')
            # pause for a freakishly shot amount of time. We need a pause so that it'll trigger a graph update
            plt.pause(0.00000000000000000001)
            # set the previous (x, z) to the current (x, z)
            px, pz, py = x, z, y
            pktid = telemetry.pkt_id
            if csvheader:
                # csvwriter.writerow(telemetry.__dict__.keys())
                csvwriter.writerow(["track_id", "x", "z", "y", "speed", "rpm"])
                csvheader = False
            csvwriter.writerow(
                [args.track, telemetry.position_x, telemetry.position_z, telemetry.position_y, telemetry.speed,
                 telemetry.rpm])
            carSpeed = telemetry.speed
            printAt('{:5.0f}'.format(telemetry.car_code), 3, 36, reverse=1)  # car id
            printAt('{:3.0f}'.format(telemetry.throttle / 2.55), 5, 11)  # throttle
            printAt('{:5.0f}'.format(telemetry.rpm), 5, 25)  # rpm
            printAt('{:7.1f}'.format(carSpeed * 3.6), 5, 47)  # speed kph
            printAt('{:>10}'.format(pktid), 3, 50, reverse=1)  # packet id
            printAt('{:4.1f}'.format(telemetry.position_x), 4, 4)  # X
            printAt('{:4.1f}'.format(telemetry.position_y), 4, 14)  # X
            printAt('{:4.1f}'.format(telemetry.position_z), 4, 24)  # X

        if pknt > 100:
            send_hb(s)
            pknt = 0
    except Exception as e:
        # printAt('Exception: {}'.format(e), 41, 1, reverse=1)
        # send_hb(s)
        # pknt = 0
        pass

    sys.stdout.flush()
