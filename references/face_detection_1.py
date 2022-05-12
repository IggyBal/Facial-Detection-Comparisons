# This is how you run the file
# python3 face_detection_1.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
# [NOTE] You must be on the same folder/directory of the prototxt and the model to run this

import numpy as np
import argparse
import imutils
import time
import cv2
from imutils.video import VideoStream

prototxt = "./deploy.prototxt.txt"
model = "./res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFORMATION] Starting Video Stream ...")
#my webcam is device 1 thus my src=1, but try running it with src=0 if you have any problems
vs = VideoStream(src=0).start()
time.sleep(2)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        #get confidence here
        thresh = 0.5
        confidence = detections[0, 0 , i, 2]
        if confidence < thresh:
            continue
        
        #box that tracks face
        box  = detections[0,0,i,3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        #draw box
        text = "{:2f}".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,225), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)

    cv2.imshow("Caffe Face Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows
vs.stop()

