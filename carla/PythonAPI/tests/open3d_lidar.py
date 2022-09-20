import numpy as np
import open3d as o3d
import cv2 as cv2
import time

def main():
    step_start = time.time()
    count = 0
    while True:
        frame_time = time.time() - step_start
        if (frame_time >= 1) and frame_time <= 1.001:
            step_start = time.time()
            count += 1
            print(count)


if __name__ == "__main__":
    main()