import glob
import cv2
import pandas
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split



def delete_ratio(annotations, images, value_min, value_max, x):
    new_annotations = []
    new_images = []

    annotations_steer = []
    for annotation in annotations:
        annotations_steer.append(annotation[1])
        
    number_to_delete = np.round(sum(map(lambda i: ((i >= value_min) and (i <= value_max)), annotations_steer)) * x)
    eliminate = random.sample([i for i, val in enumerate(annotations_steer) if ((val >= value_min) and (val <= value_max))], int(number_to_delete))
    for index, (annotations_val, images_val) in enumerate(zip(annotations, images)):
        if index in eliminate:
            continue
        else:
            new_annotations.append(annotations_val)
            new_images.append(images_val)

    return new_annotations, new_images


def get_images(list_images, type_image, image_shape):
    image_shape = (image_shape[0], image_shape[1])
    # Read the images
    array_imgs = []
    for name in list_images:
        print(name)
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if type_image == 'crop':
            img = img[240:-1, :]
        img = cv2.resize(img, image_shape)
        array_imgs.append(img)

    return array_imgs


def parse_csv(csv_data):
    array = []
    linear_speeds = csv_data['throttle'].tolist()
    angular_speeds = csv_data['steer'].tolist()
    for x, linear_speed in enumerate(linear_speeds):
        array.append((float(linear_speed), float(angular_speeds[x])))
    return array


def flip_images(images, array_annotations):
    flipped_images = []
    flipped_annotations = []
    for i, image in enumerate(images):
        flipped_images.append(cv2.flip(image, 1))
        flipped_annotations.append((array_annotations[i][0], -array_annotations[i][1]))

    images += flipped_images
    array_annotations += flipped_annotations
    return images, array_annotations


def add_extreme_data(images, array_annotations):
    for i in range(0, len(array_annotations)):
        if abs(array_annotations[i][1]) >= 0.1:
            if abs(array_annotations[i][1]) >= 0.4:
                num_iter = 25
            elif abs(array_annotations[i][1]) >= 0.3:
                num_iter = 6
            elif abs(array_annotations[i][1]) >= 0.21:
                num_iter = 1
            elif abs(array_annotations[i][1]) >= 0.15:
                num_iter = 6
            elif abs(array_annotations[i][1]) >= 0.1:  # Crece mucho
                num_iter = 10
            elif abs(array_annotations[i][1]) >= 0.03:
                num_iter = 1000
            else:
                num_iter = 0
            for j in range(0, num_iter):
                array_annotations.append(array_annotations[i])
                images.append(images[i])

    return images, array_annotations



def add_extreme_data_fr_bad(images, array_annotations):
    for i in range(0, len(array_annotations)):
        if abs(array_annotations[i][1]) >= 0.1:
            if abs(array_annotations[i][1]) >= 0.4:
                num_iter = 30
            elif abs(array_annotations[i][1]) >= 0.3:
                num_iter = 25
            elif abs(array_annotations[i][1]) >= 0.2:
                num_iter = 10
            elif abs(array_annotations[i][1]) >= 0.1:
                num_iter = 15
            elif abs(array_annotations[i][1]) >= 0.05:
                num_iter = 250
            else:
                num_iter = 110
            for j in range(0, num_iter):
                array_annotations.append(array_annotations[i])
                images.append(images[i])

    return images, array_annotations


def add_extreme_data_fl_bad(images, array_annotations):
    for i in range(0, len(array_annotations)):
        if abs(array_annotations[i][1]) >= 0.05:
            if abs(array_annotations[i][1]) >= 0.9:
                num_iter = 7
            elif abs(array_annotations[i][1]) >= 0.5:
                num_iter = 75
            elif abs(array_annotations[i][1]) >= 0.25:
                num_iter = 135
            elif abs(array_annotations[i][1]) >= 0.15:
                num_iter = 65
            elif abs(array_annotations[i][1]) >= 0.1:
                num_iter = 45
            else:
                num_iter = 22
            for j in range(0, num_iter):
                array_annotations.append(array_annotations[i])
                images.append(images[i])

    return images, array_annotations


def compute_image_annotations(id, path_to_data, type_image, img_shape, data_type):
    id = '_' + str(id)
    name_file = path_to_data + id + '/data.csv'
    dir_images = path_to_data + id + '/'
    list_images = glob.glob(dir_images + '*')
    new_list_images = []
    for image in list_images:
        # print(image)
        if image != path_to_data + id + '/data.csv':
            new_list_images.append(image)
    list_images = new_list_images

    if 'Town' in list_images[0].split('/')[2]:
        images_paths = sorted(list_images, key=lambda x: int(x.split('/')[4].split('.png')[0]))
    elif 'weird' in list_images[0].split('/')[1]:
        images_paths = sorted(list_images, key=lambda x: int(x.split('/')[3].split('.png')[0]))
    else:
        images_paths = sorted(list_images, key=lambda x: int(x.split('/')[2].split('.png')[0]))

    array_annotations = pandas.read_csv(name_file)
    array_annotations = parse_csv(array_annotations)

    images = get_images(images_paths, type_image, img_shape)
    images, array_annotations = flip_images(images, array_annotations)
    if data_type == 'extreme':
        images, array_annotations = add_extreme_data(images, array_annotations)

    array_annotations_throttle = []
    array_annotations_steer = []
    for annotation in array_annotations:
        array_annotations_throttle.append(annotation[0])
        array_annotations_steer.append(annotation[1])

    # START NORMALIZE DATA
    array_annotations_throttle = np.stack(array_annotations_throttle, axis=0)
    array_annotations_throttle = array_annotations_throttle.reshape(-1, 1)

    array_annotations_steer = np.stack(array_annotations_steer, axis=0)
    array_annotations_steer = array_annotations_steer.reshape(-1, 1)

    normalized_x = np.interp(array_annotations_throttle, (0, 1), (0, 1))
    normalized_y = np.interp(array_annotations_steer, (-1, 1), (-1, 1))

    """normalized_x = array_annotations_throttle
    normalized_y = array_annotations_steer"""

    normalized_annotations = []
    for i in range(0, len(normalized_x)):
        normalized_annotations.append([normalized_x.item(i), normalized_y.item(i)])

    array_annotations = normalized_annotations

    """for idx, image in enumerate(images):
        image_path = 'out/'
        file_name = '%d.png' % (idx)
        j = Image.fromarray(image)
        j.save(image_path + file_name)"""

    return images, array_annotations


def get_images_and_annotations(path_to_data, type_image, img_shape, data_type):
    list_dataset = glob.glob(path_to_data + '*_*')

    array_imgs = []
    array_annotations = []

    # for id in range(0, len(list_dataset)):
    # for id in range(0, 1):
    counter = 0
    for id in range(0, 10):
        print(counter)
        # id = data.split('_')[3]
        images, annotations = compute_image_annotations(id, path_to_data, type_image, img_shape, data_type)
        array_imgs += images
        array_annotations += annotations
        counter += 1

    list_town_curves = glob.glob(path_to_data + 'curves/*02')
    for path_towns in list_town_curves:
        list_curves = glob.glob(path_towns + '/*_*')
        print(path_towns)
        for data in list_curves:
            id = data.split('_')[3]
            images, annotations = compute_image_annotations(id, path_towns + '/', type_image, img_shape, data_type)
            array_imgs += images
            array_annotations += annotations

    list_weird_start = glob.glob(path_to_data + 'weird/*')
    for data in list_weird_start:
        id = data.split('_')[3]
        images, annotations = compute_image_annotations(id, path_to_data + 'weird/', type_image, img_shape, data_type)
        array_imgs += images
        array_annotations += annotations

    # Delete percentage of element with a given value
    array_annotations, array_imgs = delete_ratio(array_annotations, array_imgs, -0.01, 0.01, 0.9)

    print(len(array_imgs))
    print(len(array_annotations))

    plt.hist(np.array(array_annotations)[:,1],bins=50)
    plt.show()

    return array_imgs, array_annotations


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


def process_dataset(path_to_data, type_image, data_type, img_shape):
    array_imgs, array_annotations = get_images_and_annotations(path_to_data, type_image, img_shape, data_type)
    images_train, annotations_train, images_validation, annotations_validation = separate_dataset_into_train_validation(
        array_imgs, array_annotations)

    return images_train, annotations_train, images_validation, annotations_validation
