import os
import cv2
import sys
import glob
import pandas
import argparse
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# Display
from PIL import Image

from gradcam import GradCAM

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", action='append', help="Directory to find Data")
    args = parser.parse_args()
    return args


def main():
    """args = parse_args()
    path = args.dir[0] + '*'"""
    velocity = 13
    model = load_model('20230107-205705_rgb_brakingsimple_71_58k.h5')
    decode_predictions = keras.applications.xception.decode_predictions
    print(model.summary())
    
    image = cv2.imread("485.png")
    image = image[230:-1,:] # 280
    model.predict(np.ones((1, 66, 200, 4)))

    tensor_img = cv2.resize(image, (200, 66))/255.0

    velocity_normalize = np.interp(velocity, (0, 100), (0, 1))
    velocity_dim = np.full((66, 200), velocity_normalize)
    velocity_tensor_image = np.dstack((tensor_img, velocity_dim))
    final_image = velocity_tensor_image[np.newaxis]

    preds = model.predict(final_image)
    # GradCAM from image
    cam = GradCAM(model, 1)
    heatmap = cam.compute_heatmap(np.swapaxes(final_image, 1, 2))
    heatmap = cv2.resize(heatmap, (heatmap.shape[1], heatmap.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, tensor_img.astype(heatmap.dtype), alpha=0.5)
    e = Image.fromarray(output)
    e.save("gradcam.png")


if __name__ == "__main__":
    # execute only if run as a script
    main()