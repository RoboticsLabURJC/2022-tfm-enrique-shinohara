import numpy as np
import open3d as o3d
import cv2 as cv2

def main():
    image = cv2.imread('carla_snapshot.png')

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    YELLOW_MIN = np.array([20, 50, 50],np.uint8)
    YELLOW_MAX = np.array([36, 255, 255],np.uint8)
    frame_threshed = cv2.inRange(hsv, YELLOW_MIN, YELLOW_MAX)

    cv2.imwrite("result.png", frame_threshed)

    """image = cv2.resize(image, (200, 66), interpolation = cv2.INTER_AREA)
    image = image[np.newaxis]"""

if __name__ == "__main__":
    main()