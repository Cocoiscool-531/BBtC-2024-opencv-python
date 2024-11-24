import math
import cv2
import numpy as np


class Sample:
    """
    ONLY TAKES IN REAL WORLD VALUES
    """

    def __init__(self, x, y, distance_from_center):
        self.x = x
        self.y = y
        self.distance = distance_from_center


image_path = 'sample-test-images/'

image_name = 'IMG_7923.JPG'

# Yellow masking
lowerb = [22, 134, 0]
upperb = [255, 255, 255]


def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
            Returns an angle in radians"""
    ang = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    return -ang.__round__(2)


def open_window():
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    def get_bound_value(index: int, lower: bool):
        bound = lowerb if lower else upperb
        return bound[index]

    def set_bound_value(index: int, lower: bool, value):
        bound = lowerb if lower else upperb
        bound[index] = value
        process_image(height=height, width=width)

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


def process_image(height, width):
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
            # used to find the position of sample
            adjusted_origen = (width // 2, int(height))
            cv2.drawContours(image, [c], 0, (0, 255, 0), 3)
            moments = cv2.moments(c)
            center = (
                int(moments['m10'] / moments['m00']),
                int(moments['m01'] / moments['m00'])
            )

            rad = angle3pt((width, height), adjusted_origen, center)
            x_d = center[0] - adjusted_origen[0]
            y_d = center[1] - adjusted_origen[1]
            distance_in_px = round(math.hypot(y_d, x_d), 2)
            polar = (distance_in_px, rad)
            print(polar)
            print(center)
            cv2.drawMarker(image, center, (0, 0, 255), markerType=cv2.MARKER_STAR,
                           markerSize=5, thickness=1, line_type=cv2.LINE_AA)
            cv2.circle(image, center, 3, (255, 255, 255), 4)
            cv2.circle(image, adjusted_origen, 3, (255, 255, 255), 4)
            cv2.circle(image, (width, height), 3, (255, 255, 255), 4)
            (rect_center, (rect_width, rect_height), angle_of_rotation) = cv2.minAreaRect(c)
            print(round(angle_of_rotation))
            cv2.putText(image, 'Angle ' + str(round(angle_of_rotation)), center,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Polar coordinates:({distance_in_px} px, {rad} radians)',
                        (center[0] - 26, center[1] + 30),
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
height, width, _ = original_image.shape
open_window()
process_image(height=height, width=width)

cv2.waitKey(0)
cv2.destroyAllWindows()
