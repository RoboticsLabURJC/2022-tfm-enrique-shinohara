import os
import time
import cv2 as cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


savedModel = load_model('20220405-115235_pilotnet_new_dataset_opencv_plus_difficult_cases_extreme_cases_support_100_epochs_new_checkpoint_monitor_cp.h5')
savedModel.summary()

image = cv2.imread('44.png')
image = cv2.resize(image, (200, 66), interpolation = cv2.INTER_AREA)
image = image[np.newaxis]

for x in range(0, 10):
    t1 = time.time()
    example_result = savedModel.predict(image)
    print(time.time() - t1)
    
    time.sleep(5)

print(example_result)
