#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

from gradcam import GradCAM
from traffic_light_detector import TLClassifier

import cv2
import time


class RLAgent(object):
    """
    Base class to define agents in CARLA
    """

    def __init__(self, vehicle):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._vehicle.get_world().get_map()
        self.w = 0
        self.v = 0
        self.b = 0
        self.velocity = 0
        self.model = load_model('20230419-155255_pilotnet_model_3_91_cp.h5')
        # self.tlc = TLClassifier()
        self.first = 0
        # self.model = load_model('20221102-095537_pilotnet_model_3_51_cp.h5', custom_objects={'tf': tf})


    def run_step(self, img, current_velocity):
        """
        Execute one step of navigation.
        :return: control
        """
        cropped = img[230:-1,:] # 280
        
        """hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
        YELLOW_MIN = np.array([15, 80, 80],np.uint8)
        YELLOW_MAX = np.array([35, 255, 255],np.uint8)
        frame_threshed = cv2.inRange(hsv, YELLOW_MIN, YELLOW_MAX)
        tensor_img = cropped
        tensor_img[frame_threshed>0]=(255,0,0)"""
        # cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        tensor_img = cv2.resize(cropped, (200, 66))/255.0

        velocity_normalize = np.interp(self.velocity, (0, 100), (0, 1))
        velocity_dim = np.full((66, 200), velocity_normalize)
        velocity_tensor_image = np.dstack((tensor_img, velocity_dim))
        final_image = velocity_tensor_image[np.newaxis]

        # GradCAM from image
        """cam = GradCAM(self.model, 0)
        heatmap = cam.compute_heatmap(np.swapaxes(final_image, 1, 2))
        heatmap = cv2.resize(heatmap, (heatmap.shape[1], heatmap.shape[0]))
        (heatmap, output) = cam.overlay_heatmap(heatmap, tensor_img, alpha=0.5)"""
        
        steer_val = self.w
        throttle_val = self.v
        brake_val = self.b
        self.velocity = current_velocity
        # if data.frame_number % 10 == 0:7
        
        example_result = self.model.predict(final_image)
        # Follow lane steering valuesexpected shape changed but input image is the same as expected
        steer_val = np.interp(example_result[0][1], (0, 1), (-1, 1))
        throttle_brake_val = np.interp(example_result[0][0], (0, 1), (-1, 1))
        if throttle_brake_val >= 0: # throttle
            throttle_val = throttle_brake_val
            brake_val = 0
        else: # brake
            throttle_val = 0
            brake_val = -1*throttle_brake_val

        if self.first < 100:
            throttle_val = 0.5
            brake_val = 0.0
            self.first += 1

        self.v = throttle_val
        self.w = steer_val

        """boxes = self.tlc.detect_multi_object(img,score_threshold=0.2)
        if len(boxes) != 0:
            print("SEMAFORO")"""

        return steer_val, throttle_val, brake_val, img# , output


    def run_step_old(self, img):
        """
        Execute one step of navigation.
        :return: control
        """
        cropped = img[240:-1,:] # 280

        """hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
        YELLOW_MIN = np.array([15, 80, 80],np.uint8)
        YELLOW_MAX = np.array([35, 255, 255],np.uint8)
        frame_threshed = cv2.inRange(hsv, YELLOW_MIN, YELLOW_MAX)
        tensor_img = cropped
        tensor_img[frame_threshed>0]=(255,0,0)"""
        # cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        tensor_img = cv2.resize(cropped, (200, 66))/255.0
        final_image = tensor_img[np.newaxis]
        
        steer_val = self.w
        throttle_val = self.v
        brake_val = self.b
        # if data.frame_number % 10 == 0:
        example_result = self.model.predict(final_image)
        # Follow lane steering values
        steer_val = np.interp(example_result[0][1], (0, 1), (-1, 1))
        throttle_brake_val = np.interp(example_result[0][0], (0, 1), (-1, 1))
        if throttle_brake_val >= 0: # throttle
            throttle_val = throttle_brake_val
            brake_val = 0
        else: # brake
            throttle_val = 0
            brake_val = -1*throttle_brake_val

        if self.first < 100:
            throttle_val = 0.5
            self.first += 1

        self.v = throttle_val
        self.w = steer_val

        return steer_val, throttle_val, brake_val, img, tensor_img
