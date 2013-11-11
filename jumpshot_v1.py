#!/usr/bin/env python

import cv2
import numpy as np

cv2.namedWindow('stream', cv2.CV_WINDOW_AUTOSIZE)
vidcap = cv2.VideoCapture(1)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

first_loop = True

num_feat = 100

while(1):
    _, image = vidcap.read()
    output = image.copy()
    mask = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minSize = (50,50))
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h),(255,255,0),2)
        cv2.rectangle(mask, (x,y), (x+w,y+h), 255, -1)
    if first_loop == False:
        features = cv2.goodFeaturesToTrack(gray, num_feat, 0.01, 0.01, mask = mask)
        if features is not None:
            nxt_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, gray, features)
            for i, point in enumerate(features):
                start = tuple(point[0])
                end = tuple(nxt_pts[i,0])
                if start[1] - end[1] > 10:
                    color = (0, 0, 255)
                elif end[1] - start[1] > 10:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)
                cv2.line(image, start, end, color)
                cv2.circle(image, start, 5, color)
    else:
        first_loop = False
    prev_img = gray    
    cv2.imshow('stream', image)
    c = cv2.waitKey(10)
    if c >= 0:
        break

cv2.destroyAllWindows()
