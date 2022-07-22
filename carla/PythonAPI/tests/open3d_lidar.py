import numpy as np
import open3d as o3d
import cv2 as cv2

def main():
    image = cv2.imread('_out/004312.png')

    image = image[246:-1,:]
    image = cv2.resize(image, (200, 66), interpolation = cv2.INTER_AREA)
    tensor_image = image[np.newaxis]

    print(tensor_image.shape)

    cv2.imwrite("result.png", image)

    """image = cv2.resize(image, (200, 66), interpolation = cv2.INTER_AREA)
    image = image[np.newaxis]"""

if __name__ == "__main__":
    main()