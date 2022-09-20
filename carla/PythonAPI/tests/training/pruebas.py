import glob
from locale import normalize
import cv2
import pandas as pd
import matplotlib.pyplot as plt


from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import numpy as np


# Test
x2 = [26, 29, 48, 64, 6, 5,
      36, 66, 72, 40]
 
y2 = [26, 29, 48, 64, 6, 5,
      36, 62    , 73, 41]

 
plt.scatter(x2, y2, c ="green",
            linewidths = 2,
            marker =".",
            s = 50)
 
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()