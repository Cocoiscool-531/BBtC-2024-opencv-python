import cv2
import numpy as np

def displayImage(imgToDisplay):
    windowName = 'image'
    cv2.namedWindow('image')
    imgHeight, imgWidth, imgChannels = imgOriginal.shape
    aspectRatio = imgHeight / imgWidth
    imgResize = cv2.resize(imgToDisplay, (960, int(np.round(960 * aspectRatio, decimals=0))))
    cv2.imshow(windowName, imgResize)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


imagePath = 'C:\\Users\\Tim\\Desktop\\FTC\\2024-2025 Into the Deep\\ML work\\Yellow test set for angle\\'
imgName = 'center_70'
imgType = '.jpg'

imgOriginal = cv2.imread(imagePath+imgName+imgType) # BGR format
viewWindowStart = (470, 420)
viewWindowEnd = (868, 868)
#imgOriginal = imgOriginal[viewWindowStart[1]:viewWindowEnd[1], viewWindowStart[0]:viewWindowEnd[0]]

hsvImage = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

# Yellow masking
yellowMask = cv2.inRange(hsvImage,(5, 169, 109), (31, 255, 255))
kernel = np.ones((5, 5), np.uint8)
yellowMask = cv2.dilate(yellowMask, kernel, iterations=2)
yellowMask = cv2.erode(yellowMask, kernel, iterations=1)

yellowMaskApplied = cv2.bitwise_and(imgOriginal,imgOriginal,mask = yellowMask)


contours, _ = cv2.findContours(yellowMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    if cv2.contourArea(c) > 15000:
        print('Found contour')
        cv2.drawContours(imgOriginal,[c],0,(0,255,0),3)
        M = cv2.moments(c)
        Mx = int(M['m10'] / M['m00'])
        My = int(M['m01'] / M['m00'])
        cv2.drawMarker(imgOriginal, (Mx, My), (0, 0, 255), markerType=cv2.MARKER_STAR,
                       markerSize=10, thickness=2, line_type=cv2.LINE_AA)
        (rectCenter, (rectWidth, rectHeight), angleOfRotation) = cv2.minAreaRect(c)
        print(round(angleOfRotation))
        imgOriginal = cv2.putText(imgOriginal,'Angle ' + str(round(angleOfRotation)) + ' deg',(0,150),
                                  cv2.FONT_HERSHEY_SIMPLEX,5,(0, 255, 255),2,cv2.LINE_AA)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        imgOriginal = cv2.drawContours(imgOriginal, [box], 0, (0, 255, 255), 2)

displayImage(imgOriginal)


# Red Masking
#rmask1 = cv2.inRange(hsv, (166, 143, 60), (180, 240, 240))
#rmask2 = cv2.inRange(hsv, (0, 143, 60), (3, 240, 240))

#redmask = rmask1 + rmask2
#cv2.imwrite(imagePath + imgName + '_redmask' + imgType, redmask)

# Red Contour Detection
#_, thresh = cv2.threshold(redmask, 127, 255, cv2.THRESH_BINARY)  # Threshold to get binary image
#contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#mask = np.zeros_like(thresh)
#cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

#kernel = np.ones((5, 5), np.uint8)  # adjust this for dialation size
#dilated_mask = cv2.dilate(mask, kernel, iterations=1)

#dilated_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#min_area = 10000  # 10000 is decent
#filtered_contours = [cnt for cnt in dilated_contours if cv2.contourArea(cnt) > min_area]

# Draw the final contours on the original image or mask
#dialtedfiltered = cv2.drawContours(src, filtered_contours, -1, (255, 0, 0), 5)
#cv2.imwrite(f".\\image-{i}\\dialted_cntr.jpg", dialtedfiltered)

# get bounding boxes
#boxes = [cv2.boundingRect(cnt) for cnt in filtered_contours]
#boxed = src.copy()
#for x, y, w, h in boxes:
#    cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0) if h / w >= 0.7 else (50, 50, 255), 20)
# label bounding boxes with area of contour
#for k, (x, y, w, h) in enumerate(boxes):
#    cv2.putText(boxed, f"{int(cv2.contourArea(filtered_contours[k]))}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 3,
#                (250, 250, 180), 12)
#cv2.imwrite(f".\\image-{i}\\bounding_boxes.jpg", boxed)

# Yellow Masking
#yellowmask = cv2.inRange(hsv, (5, 169, 109), (31, 255, 255))

#cv2.imwrite(f".\\image-{i}\\yellowmask.jpg", yellowmask)

#ylwapplied = cv2.bitwise_and(src, src, mask=yellowmask)
#redapplied = cv2.bitwise_and(src, src, mask=redmask)
#cv2.imwrite(f".\\image-{i}\\yellowmask-applied.jpg", ylwapplied)
#cv2.imwrite(f".\\image-{i}\\redmask-applied.jpg", redapplied)