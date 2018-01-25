import cv2
import dlib
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from trackersFunc import setTracker as tr
from trackersFunc import overlaps as olap



class VideoCamera(object):



    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture("a.mp4")
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

        op = True
        tracker = []
        success, image = self.video.read()
        (h, w) = image.shape[:2]

        # [Shape of image is accessed by img.shape. It returns a tuple of number
        # of rows, columns and channels (if image is color):

        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),  # convert frame into a blob
                                     0.007843, (300, 300), 127.5)

        net.setInput(blob)  # me blob eka nural network ekata deela eke detections gannawa
        detections = net.forward()  # forward method eken return wenne <type 'numpy.ndarray'>.

        coordinates = []
        print "detections"
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):  # shape eken return wenne array eke dimention eka shape[2] kiyanne eke 3weni attribute
            #  eka detect kragatta objects gana
            # take the probability of each detection
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence given
            if confidence > 0.2:
                #
                idx = int(detections[0, 0, i, 1])  # get the index from detected object
                # 1 wenne i kiyana object eke wargaya class eke position eka
                if (idx == 15):  # filter only humans
                    box = detections[0, 0, i, 3:7] * np.array(
                        [w, h, w, h])  ##3:7 wenne i kiyana object eke x1=3,y1=4,x2=5,y2=6 points 4

                    (startX, startY, endX, endY) = box.astype("int")
                    # print [startX, startY, endX, endY],
                    coordinates.append([startX, startY, endX, endY])
                    # draw the prediction on the frame
                    # label = "{}: {:.2f}%".format(CLASSES[idx],
                    # confidence * 100)'
                    label = "person"

                    cv2.rectangle(image, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                    yyy = startY - 15 if startY - 15 > 15 else startY + 15
                    #cv2.putText(image, label, (startX, yyy),
                     #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        print coordinates
        if (op):
            op = False
            tracker = [dlib.correlation_tracker() for _ in xrange(len(coordinates))]
            # Provide the tracker the initial position of the object
            [tracker[i].start_track(image, dlib.rectangle(*rect)) for i, rect in enumerate(coordinates)]
            # tracker = dlib.correlation_tracker()
            # tracker.start_track(frame, dlib.rectangle(startX,startY,endX,endY))
            # = [dlib.correlation_tracker() for _ in xrange(len(coordinates))]
            # Provide the tracker the initial position of the object
            # [tracker[i].start_track(frame, dlib.rectangle(*rect)) for i, rect in enumerate(coordinates)]
        print
        print "trackers"
        trcks = []
        for ii in xrange(len(tracker)):
            tracker[ii].update(image)
            # Get the position of th object, draw a
            # bounding box around it and display it.
            rect = tracker[ii].get_position()
            pt1 = (int(rect.left()), int(rect.top()))
            pt2 = (int(rect.right()), int(rect.bottom()))
            cv2.rectangle(image, pt1, pt2, (255, 255, 255), 3)
            # print [int(rect.left()), int(rect.top()),int(rect.right()), int(rect.bottom())],
            trcks.append([int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())])
            label = "tl" + str(ii)
            # Get the position of the object, draw a
            # bounding box around it and display it.
            y = pt1[1] - 15 if pt1[1] - 15 > 15 else pt1[1] + 15
            cv2.putText(image, label, (pt1[0], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        print trcks
        # SI = max(0, min(XA2, XB2) - Max(XA1, XB1)) * Max(0, Min(YA2, YB2) - Max(YA1, YB1))

        olap(trcks, coordinates, image,tracker)
        print
        # show the output frame

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop


        # update the FPS counter


        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()