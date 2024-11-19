import cv2
import apriltag
import argparse
import numpy as np

image_path = 'sample-test-images/'
image_name = 'AprilTagTest.jpg'

image = cv2.imread(image_path + image_name)
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)
results = detector.detect(grey)
if len(results) > 1:
    print(str(len(results)), "tags detected")
else:
    print(str(len(results)), "tag detected")

for r in results:

    (ptA, ptB, ptC, ptD) = r.corners
    ptB = int(ptB[0]), int(ptB[1])
    ptC = int(ptC[0]), int(ptC[1])
    ptD = int(ptD[0]), int(ptD[1])
    ptA = int(ptA[0]), int(ptA[1])

    cv2.line(image, ptA, ptB, (0, 255, 0), 2)
    cv2.line(image, ptB, ptC, (0, 255, 0), 2)
    cv2.line(image, ptC, ptD, (0, 255, 0), 2)
    cv2.line(image, ptD, ptA, (0, 255, 0), 2)

    (cX, cY) = (int(r.center[0]), int(r.center[1]))
    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

    tagFamily = r.tag_family.decode("utf-8")
    cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print("tagFamily: {}".format(tagFamily))

    cv2.imshow("AprilTag", image)
    cv2.waitKey(0)