from flask import Flask, Response
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time
import supervision as sv

app = Flask(__name__)
picam2 = Picamera2()

# Camera setup
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(2)  # Let the camera settle

# Load NCNN model
model = YOLO("tube_detection/weights/best_ncnn_model")

# Initialize ByteTrack tracker
tracker = sv.ByteTrack(
    track_activation_threshold=0.15,
    lost_track_buffer=50,
    minimum_matching_threshold=0.8,
    frame_rate=30  # Adjust based on Pi Camera FPS
)

# Parameters
aspect_ratio_threshold = 0.4
block_time_threshold = 2.0
track_data = {}

# Streaming generator
def generate():
    while True:
        frame = picam2.capture_array()
        current_time = time.time()

        # YOLOv8-NCNN Inference
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        tracked = tracker.update_with_detections(detections)

        blocked_warning = False
        for tracker_id, bbox in zip(tracked.tracker_id, tracked.xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            width, height = x2 - x1, y2 - y1
            aspect_ratio = width / height

            if tracker_id not in track_data:
                track_data[tracker_id] = {
                    "aspect_ratio": aspect_ratio,
                    "block_time": 0,
                    "start_time": current_time
                }
            else:
                if aspect_ratio > aspect_ratio_threshold:
                    elapsed = current_time - track_data[tracker_id]["start_time"]
                    track_data[tracker_id]["block_time"] += elapsed
                else:
                    track_data[tracker_id]["block_time"] = 0
                track_data[tracker_id]["start_time"] = current_time

            is_blocked = track_data[tracker_id]["block_time"] >= block_time_threshold
            label = "BLOCKED" if is_blocked else f"Tube {tracker_id}"
            color = (0, 0, 255) if is_blocked else (0, 255, 0)
            if is_blocked:
                blocked_warning = True

            # Draw annotated bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if blocked_warning:
            cv2.putText(frame, "BLOCKED", (frame.shape[1] - 200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame for MJPEG
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
