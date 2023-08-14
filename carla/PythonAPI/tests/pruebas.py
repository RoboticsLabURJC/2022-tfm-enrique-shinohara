import os
import cv2
import sys
import glob
import pandas
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# Display
from PIL import Image

from gradcam import GradCAM
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", action='append', help="Directory to find Data")
    args = parser.parse_args()
    return args

def separate_dataset_into_train_validation(array_x, array_y):
    images_train, images_validation, annotations_train, annotations_validation = train_test_split(array_x, array_y,
                                                                                                  test_size=0.30,
                                                                                                  random_state=42,
                                                                                                  shuffle=True)

    print('Images train -> ' + str(len(images_train)))
    print('Images validation -> ' + str(len(images_validation)))
    print('Annotations train -> ' + str(len(annotations_train)))
    print('Annotations validation -> ' + str(len(annotations_validation)))
    # Adapt the data
    images_train = np.stack(images_train, axis=0)
    annotations_train = np.stack(annotations_train, axis=0)
    images_validation = np.stack(images_validation, axis=0)
    annotations_validation = np.stack(annotations_validation, axis=0)

    print('Images train -> ' + str(images_train.shape))
    print('Images validation -> ' + str(images_validation.shape))
    print('Annotations train -> ' + str(annotations_train.shape))
    print('Annotations validation -> ' + str(annotations_validation.shape))

    return images_train, annotations_train, images_validation, annotations_validation


def delete_until(annotations, images, x, counter, bins, type):
    new_annotations = []
    new_images = []

    annotations_steer = []
    for annotation in annotations:
        annotations_steer.append(annotation[type])

    total_elimination = []
    for index, counts in enumerate(counter):
        if counts > x:
            number_to_delete = counts - x
            eliminate = random.sample([i for i, val in enumerate(annotations_steer) if ((val >= bins[index]) and (val <= bins[index+1]))], int(number_to_delete))
            total_elimination.extend(eliminate)

    for index, (annotations_val, images_val) in enumerate(zip(annotations, images)):
        if index in total_elimination:
            continue
        else:
            new_annotations.append(annotations_val)
            new_images.append(images_val)

    return new_annotations, new_images


def main():
    """args = parse_args()
    path = args.dir[0] + '*'"""

    if os.path.exists('y_predicted_steering.npy'):
            print("LOADING FROM DATA")
            x_true = np.load('x_true_steering.npy', allow_pickle=True)
            y_predicted = np.load('y_predicted_steering.npy', allow_pickle=True)
            x_true_throttle = np.load('x_true_throttle.npy', allow_pickle=True)
            y_predicted_throttle = np.load('y_predicted_throttle.npy', allow_pickle=True)

    else:
        array_imgs = []
        array_annotations = []
        if os.path.exists('array_imgs.npy'):
            array_imgs = np.load('array_imgs.npy', allow_pickle=True)
            array_annotations = np.load('array_annotations.npy', allow_pickle=True)

        (n2, bins2, patches) = plt.hist(np.array(array_annotations)[:,1],bins=50)
        # # Delete until reach a certain max value
        array_annotations, array_imgs = delete_until(array_annotations, array_imgs, 40000, n2, bins2, 1)

        (n2, bins2, patches) = plt.hist(np.array(array_annotations)[:,0],bins=50)
        # Delete until reach a certain max value
        array_annotations, array_imgs = delete_until(array_annotations, array_imgs, 30000, n2, bins2, 0)

        images_train, annotations_train, images_validation, annotations_validation = separate_dataset_into_train_validation(
            array_imgs, array_annotations)

        annotations_val = annotations_validation
        images_val = images_validation
        model = load_model('20230517-094715_pilotnet_model_3_15+101_cp.h5')
        print(model.summary())
        
        """image = cv2.imread("485.png")
        image = image[200:-1,:] # 280
        model.predict(np.ones((1, 66, 200, 4)))

        tensor_img = cv2.resize(image, (200, 66))/255.0

        velocity_normalize = np.interp(velocity, (0, 100), (0, 1))
        velocity_dim = np.full((66, 200), velocity_normalize)
        velocity_tensor_image = np.dstack((tensor_img, velocity_dim))
        final_image = velocity_tensor_image[np.newaxis]"""

        # Scatter plot of the steering values
        x_true = []
        y_predicted = []

        # Scatter plot of the throttle values
        x_true_throttle = []
        y_predicted_throttle = []
        for x in (range(0, len(annotations_val), 50)):
            print(x)
            x_true.append(annotations_val[x][1])
            x_true_throttle.append(annotations_val[x][0])

            final_image = images_val[x] / 255.0
            print(annotations_val[x])
            velocity_dim = np.full((66, 200), annotations_val[x][2])
            final_image = np.dstack((final_image, velocity_dim))

            final_image = final_image[np.newaxis]
            prediction = model.predict(final_image)

            y_predicted.append(prediction[0][1])
            y_predicted_throttle.append(prediction[0][0])

        np.save('x_true_steering.npy', x_true, allow_pickle=True)
        np.save('y_predicted_steering.npy', y_predicted, allow_pickle=True)
        np.save('x_true_throttle.npy', x_true_throttle, allow_pickle=True)
        np.save('y_predicted_throttle.npy', y_predicted_throttle, allow_pickle=True)

    # Steering values plot
    fig1, ax1 = plt.subplots(figsize=(20, 10))
    ax1.plot(x_true, c ="green")
    ax1.plot(y_predicted, c ="red")

    plt.xlabel("instance")
    plt.ylabel("steering")
    name = 'plot_graph_steering'
    fig1.savefig(name, dpi = 100)
    plt.close(fig1)

    # Throttle values plot
    fig1, ax1 = plt.subplots(figsize=(20, 10))
    ax1.plot(x_true_throttle, c ="green")
    ax1.plot(y_predicted_throttle, c ="red")

    plt.xlabel("instance")
    plt.ylabel("throttle")
    name = 'plot_graph_throttle'
    fig1.savefig(name, dpi = 100)
    plt.close(fig1)


if __name__ == "__main__":
    # execute only if run as a script
    main()