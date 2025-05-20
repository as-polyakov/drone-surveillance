import datetime
import logging
import datetime as dt
import signal
from time import sleep

import cv2 as cv

# plan:
# shoot the footage every 5 sec
# analyze the picture for smoke or fire
# save the image to local folder with some annotation in case of detection
# separate process (rclone move) sends saved images to remote folder


cascPath = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
objectsCascade = cv.CascadeClassifier(cascPath)
logger = logging.getLogger(__name__)
logging.basicConfig(filename='webcam.log', level=logging.INFO)


def generate_filename(prefix='file', extension='txt', dt_format='%Y%m%d_%H%M%S'):
    timestamp = datetime.datetime.now().strftime(dt_format)
    return f"{prefix}_{timestamp}.{extension}"


def capture_frame():
    # cv.namedWindow("test")
    keep_processing = True

    cam = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)

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
            detect_objects(frame)
            if frame_no % 10 == 0:
                cv.imwrite(generate_filename('opencv', '.png'), frame)
        sleep(1)
        frame_no += 1

def detect_objects(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    detectedObjects = objectsCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the detectedObjects
    for (x, y, w, h) in detectedObjects:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if len(detectedObjects) > 0:
        logger.info("Found " + str(len(detectedObjects)) + " detectedObjects")
        cv.imwrite(generate_filename('objects-', ".png"), frame)


if __name__ == "__main__":
    capture_frame()
