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
from tensorflow.keras.models import load_model

import cv2


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
        self.model = load_model('20221013-202213_pilotnet_model_3_albumentations_no_crop_cp.h5')


    def run_step(self, img):
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

        tensor_img = cv2.resize(cropped, (66, 200), interpolation = cv2.INTER_AREA)
        final_image = tensor_img[np.newaxis]
        
        steer_val = self.w
        throttle_val = self.v
        # if data.frame_number % 10 == 0:
        example_result = self.model.predict(final_image)
        # Follow lane steering values
        steer_val = np.interp(example_result[0][1], (0, 35), (-1, 1))
        # print(example_result[0][1])
        # print(example_result[0][0])
        throttle_val = 0.15 # example_result[0][0]

        self.v = throttle_val
        self.w = steer_val

        return steer_val, throttle_val, img
