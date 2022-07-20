import cv2 as cv2
import sys
import numpy as np

CROPPED_START_Y = 147

img = cv2.imread('/home/yujiro/carla/CARLA_0.9.2/PythonAPI/tests/_out/000236.png')

def check_leftright_lanes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    edges = cv2.Canny(gray, 100, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (21, 2))

    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    target_x = int(img.shape[1]/2)
    target_y = 220

    left_lane_x = 0
    right_lane_x = 0
    # cv2.circle(img, (int(target_x), int(target_y)), 3, (0, 0, 255), -1)
    for x in range(target_x, img.shape[1]):
        if eroded[x, target_y] == 255:
            right_lane_x = x
    for x in reversed(range(0, target_x)):
        if eroded[x, target_y] == 255:
            left_lane_x = x

    cv2.circle(img, (int(left_lane_x), int(target_y)), 3, (0, 0, 255), -1)
    cv2.circle(img, (int(right_lane_x), int(target_y)), 3, (0, 0, 255), -1)


    return img

croped_img = check_leftright_lanes(img)

"""cv2.circle(croped_img, left, 3, (0, 0, 255), -1)
cv2.circle(croped_img, right, 3, (0, 0, 255), -1)"""

# img = np.array(cv2.resize(eroded, dsize=(640, 480), interpolation=cv2.INTER_CUBIC))

cv2.imshow("", croped_img)
cv2.waitKey(0)