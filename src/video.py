import datetime
import logging
from sys import platform
import signal
from time import sleep
import argparse
from ultralytics import YOLO
import collections
collections.MutableMapping = collections.abc.MutableMapping
from dronekit import connect, VehicleMode
import cv2 as cv

# plan:
# shoot the footage every 5 sec
# analyze the picture for smoke or fire
# save the image to local folder with some annotation in case of detection
# separate process (rclone move) sends saved images to remote folder

def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Create handlers
    file_handler = logging.FileHandler('webcam.log')
    console_handler = logging.StreamHandler()
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger;


logger = init_logger()


rtsp_url = "rtsp://root:12345@192.168.1.10:554/stream1"
is_local = (platform != "linux" and platform != "linux2")
vehicle = None

def connect_to_vehicle(connection_string):
    try:
        vehicle = connect(connection_string, wait_ready=True)
        logger.info("Connected to vehicle")
        return vehicle
    except Exception as e:
        logger.error(f"Failed to connect to vehicle: {e}")
        raise

def generate_filename(prefix='file', extension='txt', dt_format='%Y%m%d_%H%M%S'):
    timestamp = datetime.datetime.now().strftime(dt_format)
    return f"{prefix}_{timestamp}.{extension}"


def init_camera(camera_type):
    if camera_type == 'rtsp':
        return cv.VideoCapture(rtsp_url)
    elif camera_type == 'v4l':
        cam = cv.VideoCapture(0)
        cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        cam.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'RGB3'))
        return cam
    else:
        raise ValueError(f"Unsupported camera type: {camera_type}")

def signal_fire(frame):
    """Signal fire detection and log GPS coordinates"""
    if vehicle is None:
        logger.error("Vehicle not connected")
        return
    
    try:
        # Get current GPS coordinates
        lat = vehicle.location.global_frame.lat
        lon = vehicle.location.global_frame.lon
        alt = vehicle.location.global_frame.alt
        
        logger.info(f"Fire detected at coordinates: lat={lat}, lon={lon}, alt={alt}")
        print(f"Fire detected at coordinates: lat={lat}, lon={lon}, alt={alt}")
        
        # Save the image with coordinates in filename
        filename = generate_filename(f'fire-{lat:.6f}-{lon:.6f}', '.png')
        cv.imwrite(filename, frame)
        logger.info(f"Saved fire detection image to {filename}")
    except Exception as e:
        logger.error(f"Failed to get GPS coordinates: {e}")


def start_event_loop(camera_type):
    # cv.namedWindow("test")
    keep_processing = True

    cam = init_camera(camera_type)
    model = init_model()

    def exit_gracefully(signum, frame):
        nonlocal keep_processing
        keep_processing = False
        cam.release()
    def init_webcam():
        signal.signal(signal.SIGINT, exit_gracefully)
        signal.signal(signal.SIGTERM, exit_gracefully)
        logger.info("Initializing webcam")
        while not cam.isOpened():
            sleep(1)
        logger.info("Webcam initialized")

    init_webcam()
    frame_no = 0;
    while keep_processing:
        ret, frame = cam.read()
        if not ret:
            logger.warning("failed to grab frame")
        else:
            detected_class = detect_objects(model, frame)
            if detected_class == "fire":
                signal_fire(frame)

            if frame_no % 10 == 0:
                cv.imwrite(generate_filename('opencv', '.png'), frame)
        sleep(1)
        logger.log(logging.INFO, f"Lattitude: {vehicle.location.global_frame.lat}")
        frame_no += 1
        frame_no += 1

def init_model():
    classifier = YOLO("yolov8n-fire-best.pt")
    return classifier

def detect_objects(classifier, frame):
    results = classifier(source=frame, show=False)
    print(results[0].names[results[0].probs.top1])
    return results[0].names[results[0].probs.top1]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video capture with object detection')
    parser.add_argument('-d', '--device', choices=['rtsp', 'v4l'], required=True,
                      help='Camera device type: rtsp for RTSP stream or v4l for local webcam')
    parser.add_argument('-c', '--connection', default='tcp:0.0.0.0:5760',
                      help='DroneKit connection string (default: tcp:0.0.0.0:5760)')
    args = parser.parse_args()
    
    # Connect to the vehicle
    try:
        vehicle = connect_to_vehicle(args.connection)
        start_event_loop(args.device)
    except Exception as e:
        logger.error(f"Failed to start: {e}")
    finally:
        if vehicle is not None:
            vehicle.close()
