***
# Gran Turismo 7 Track Capture Tool

## Introduction

This repository contains a Gran Turismo 7 app used to capture telemetry data as it is received, in a .csv file.
It was initially made to capture car position x/z/y in order to learn tracks coordinates for further automatic recognition of the track using ML/AI.

``
PlayStation 4/5 (port 33749) -> GT7Proxy (port 33740)
``

## Running the capture

All tracks are referenced [here](https://github.com/ddm999/gt7info/blob/web-new/_data/db/course.csv)
The --track parameter should reflect the track ID you will be using to capture telemetry data.

```
usage: GT7Map2CSV.py [-h] --ps_ip PS_IP [--logpackets LOGPACKETS] --track TRACK

options:
  -h, --help            show this help message and exit
  --ps_ip PS_IP         Playstation 4/5 IP address. Accepts IP or FQDN provided it
                        resolves to something.
  --logpackets LOGPACKETS
                        Optionnaly log packets for future playback using https://git
                        hub.com/vthinsel/Python_UDP_Receiver/UDPSend_timed.py
                        .Default is False
  --track TRACK         Track ID as per https://github.com/ddm999/gt7info/blob/web-
                        new/_data/db/course.csv
```

