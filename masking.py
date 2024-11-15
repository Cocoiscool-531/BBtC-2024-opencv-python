import cv2
import numpy as np

# imagePath = 'sample-test-images/yellow-test-set-for-angle/'
image_path = 'sample-test-images/'
# imgName = 'center_70.jpg'
image_name = 'IMG_20240914_105819807.jpg'

# Yellow masking
lowerb = [5, 169, 109]
upperb = [31, 255, 255]


def open_window():
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    def get_bound_value(index: int, lower: bool):
        bound = lowerb if lower else upperb
        return bound[index]

    def set_bound_value(index: int, lower: bool, value):
        bound = lowerb if lower else upperb
        bound[index] = value
        process_image()

    def create_bound_trackbar(name, index, lower):
        cv2.createTrackbar(name, 'image',
                           get_bound_value(index, lower), 255,
                           lambda value: set_bound_value(index, lower, value))

    create_bound_trackbar('hue_lbound', 0, True)
    create_bound_trackbar('hue_ubound', 0, False)
    create_bound_trackbar('sat_lbound', 1, True)
    create_bound_trackbar('sat_ubound', 1, False)
    create_bound_trackbar('val_lbound', 2, True)
    create_bound_trackbar('val_ubound', 2, False)


def bgr_to_hsv(bgr_color):
    bgr_pixel = np.array(bgr_color, dtype=np.uint8)
    hsv_pixel = cv2.cvtColor(np.array([[bgr_pixel]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
    return hsv_pixel


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pass
    elif event == cv2.EVENT_LBUTTONUP:
        pass
    elif event == cv2.EVENT_MOUSEMOVE:
        pass

    bgr_color = original_image[y, x]
    hsv_color = bgr_to_hsv(bgr_color)
    cv2.setWindowTitle('image', f'({x},{y}) = HSV {hsv_color}')


def display_image(image):
    cv2.imshow('image', image)


def process_image():
    image = original_image.copy()
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_image, tuple(lowerb), tuple(upperb))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) > 15000:
            print('Found contour')
            cv2.drawContours(image, [c], 0, (0, 255, 0), 3)
            moments = cv2.moments(c)
            center = (
                int(moments['m10'] / moments['m00']),
                int(moments['m01'] / moments['m00'])
            )
            cv2.drawMarker(image, center, (0, 0, 255), markerType=cv2.MARKER_STAR,
                           markerSize=5, thickness=1, line_type=cv2.LINE_AA)
            (rect_center, (rect_width, rect_height), angle_of_rotation) = cv2.minAreaRect(c)
            print(round(angle_of_rotation))
            cv2.putText(image, 'Angle ' + str(round(angle_of_rotation)) + ' deg', center,
                                         cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2, cv2.LINE_AA)
            min_area_rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(min_area_rect)
            box = np.intp(box)
            cv2.drawContours(image, [box], 0, (0, 255, 255), 1)
            display_image(image)


original_image = cv2.imread(image_path + image_name)  # BGR format
height, width, channels = original_image.shape
aspect_ratio = height / width
original_image = cv2.resize(original_image, (1280, int(np.round(1280 * aspect_ratio, decimals=0))))

open_window()
process_image()

cv2.waitKey(0)
cv2.destroyAllWindows()

# Red Masking
# rmask1 = cv2.inRange(hsv, (166, 143, 60), (180, 240, 240))
# rmask2 = cv2.inRange(hsv, (0, 143, 60), (3, 240, 240))

# redmask = rmask1 + rmask2
# cv2.imwrite(imagePath + imgName + '_redmask' + imgType, redmask)

# Red Contour Detection
# _, thresh = cv2.threshold(redmask, 127, 255, cv2.THRESH_BINARY)  # Threshold to get binary image
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# mask = np.zeros_like(thresh)
# cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# kernel = np.ones((5, 5), np.uint8)  # adjust this for dialation size
# dilated_mask = cv2.dilate(mask, kernel, iterations=1)

# dilated_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# min_area = 10000  # 10000 is decent
# filtered_contours = [cnt for cnt in dilated_contours if cv2.contourArea(cnt) > min_area]

# Draw the final contours on the original image or mask
# dialtedfiltered = cv2.drawContours(src, filtered_contours, -1, (255, 0, 0), 5)
# cv2.imwrite(f".\\image-{i}\\dialted_cntr.jpg", dialtedfiltered)

# get bounding boxes
# boxes = [cv2.boundingRect(cnt) for cnt in filtered_contours]
# boxed = src.copy()
# for x, y, w, h in boxes:
#    cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0) if h / w >= 0.7 else (50, 50, 255), 20)
# label bounding boxes with area of contour
# for k, (x, y, w, h) in enumerate(boxes):
#    cv2.putText(boxed, f"{int(cv2.contourArea(filtered_contours[k]))}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 3,
#                (250, 250, 180), 12)
# cv2.imwrite(f".\\image-{i}\\bounding_boxes.jpg", boxed)

# Yellow Masking
# yellowmask = cv2.inRange(hsv, (5, 169, 109), (31, 255, 255))

# cv2.imwrite(f".\\image-{i}\\yellowmask.jpg", yellowmask)

# ylwapplied = cv2.bitwise_and(src, src, mask=yellowmask)
# redapplied = cv2.bitwise_and(src, src, mask=redmask)
# cv2.imwrite(f".\\image-{i}\\yellowmask-applied.jpg", ylwapplied)
# cv2.imwrite(f".\\image-{i}\\redmask-applied.jpg", redapplied)
