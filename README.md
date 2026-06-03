# Manufacturing-Product-Blockage-Detection
## Overview
This repository contains a real-time computer vision system designed to detect and flag plastic products that get stuck at the exit cavity of a blow molding machine. The solution is optimized for edge deployment on a Raspberry Pi 4, utilizing a custom-trained YOLOv8 model converted to the NCNN framework for low-latency inference.

When a blockage occurs, the system visually highlights the stuck product on a live video stream, which is broadcasted over a local network using a Flask web server.

## Features
* Edge-Optimized Inference: Uses a custom YOLOv8 model exported to NCNN format for accelerated performance on ARM architecture.
* Robust Object Tracking: Integrates the ByteTrack algorithm via the Supervision library to maintain object identities across frames.
* Custom Blockage Logic: Identifies stuck products by monitoring the bounding box aspect ratio and calculating the dwell time of each tracked item.
* Live Network Streaming: Streams annotated MJPEG video feeds over the local network via Flask, allowing operators to monitor the machine remotely.

## Hardware Requirements
* Raspberry Pi 4 Model B
* Raspberry Pi Camera Module V2 (or compatible PiCamera2 supported hardware)

## Software Stack
* Python 3
* Flask (Web Server & MJPEG Streaming)
* Picamera2 (Camera Interface)
* Ultralytics (YOLOv8)
* Supervision (ByteTrack & Detections)
* OpenCV (Image Processing & Annotation)

## How It Works
1. Frame Capture: The PiCamera2 captures RGB frames at a resolution of 640x480.
2. Detection: The NCNN model processes the frame to detect plastic tubes.
3. Tracking: ByteTrack assigns and maintains a unique ID for every detected tube.
4. Blockage Logic: 
   - The system calculates the width-to-height aspect ratio of each bounding box.
   - If a tracked object maintains an aspect ratio greater than 0.4 for more than 2.0 consecutive seconds, it is classified as stuck.
5. Alerting: The bounding box color shifts from green to red, the label updates to BLOCKED, and a global warning is overlaid on the video stream.

## Installation & Setup

### 1. Clone the Repository
git clone https://github.com/Bashithaperera/Manufacturing-Product-Blockage-Detection.git

cd Manufacturing-Product-Blockage-Detection

### 2. Install Dependencies
It is recommended to run this in a virtual environment.
pip install Flask picamera2 ultralytics opencv-python supervision

### 3. Model Placement
Ensure your custom-trained NCNN model weights are placed in the correct directory. By default, the script looks for the model at:
tube_detection/weights/best_ncnn_model

### 4. Execution
Run the detection script:
python detect.py

### 5. Viewing the Stream
Once the script is running, open a web browser on any device connected to the same local network and navigate to:
http://<raspberry-pi-ip-address>:5000

## Configuration
You can fine-tune the tracking and blockage logic by adjusting the parameters directly in detect.py:

* aspect_ratio_threshold = 0.4: Minimum aspect ratio to trigger the blockage timer.
* block_time_threshold = 2.0: Consecutive seconds the aspect ratio must be exceeded to flag a blockage.
* track_activation_threshold = 0.15: Confidence threshold for ByteTrack.

## Output (frame extracted from results_sample.mp4)

<img width="303" height="435" alt="image" src="https://github.com/user-attachments/assets/50f54dad-7ef9-4011-995f-3979657daf78" />


## License
MIT License
