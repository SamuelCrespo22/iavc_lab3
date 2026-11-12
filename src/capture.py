import cv2 as cv
import sys

def open_camera(source=0, width=640, height=480):
    cap = cv.VideoCapture(source)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit(1)
    return cap