import logging
import datetime as dt
from time import sleep

import cv2 as cv


cascPath = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascPath)
logger = logging.getLogger(__name__)
logging.basicConfig(filename='webcam.log',level=logging.INFO)
def capture_frame():
    #cv.namedWindow("test")
    cam = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)
    logger.info("Initializing webcam")
    while not cam.isOpened():
        sleep(1)

    logger.info("Webcam initialized")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            exit(1)
        detect_objects(frame)
        sleep(1)

    cv.imwrite('opencv' + str(1) + '.png', frame)
    cam.release()

    cv.destroyAllWindows()

def detect_objects(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if len(faces) > 0:
        print("Found " + str(len(faces)) + " faces")
        cv.imwrite('objects-' + dt.datetime.now().strftime("%d-%m-%Y-%H:%M:%S") + ".png", frame)

if __name__ == "__main__":
    capture_frame()