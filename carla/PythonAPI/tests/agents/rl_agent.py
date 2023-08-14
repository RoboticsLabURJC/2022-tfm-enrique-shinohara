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
import matplotlib.pyplot as plt

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
        # self.model = load_model('20230517-094715_pilotnet_model_3_15+101_cp.h5') # PilotNet** dataset X vehicles
        self.model = load_model('20230803-133009_pilotnet_model_3_101_cp.h5') # PilotNet** dataset 1 vehicles
        # self.model = load_model('20230804-124320_pilotnet_model_3_91_cp.h5') # PilotNet* dataset 1 vehicles
        # self.model = load_model('20221111-225504_rgb.h5')
        
        # self.tlc = TLClassifier()
        self.first = 0
        # self.model = load_model('20221102-095537_pilotnet_model_3_51_cp.h5', custom_objects={'tf': tf})


    def run_step_pilotnet_star_star(self, img, current_velocity):
        """
        Execute one step of navigation.
        :return: control
        """
        cropped = img[200:-1,:] # 240
        
        """hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
        YELLOW_MIN = np.array([15, 80, 80],np.uint8)
        YELLOW_MAX = np.array([35, 255, 255],np.uint8)
        frame_threshed = cv2.inRange(hsv, YELLOW_MIN, YELLOW_MAX)
        tensor_img = 
        
        tensor_img[frame_threshed>0]=(255,0,0)"""
        # cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        tensor_img = cv2.resize(cropped, (200, 66))/255.0
        # final_image = tensor_img[np.newaxis]

        velocity_normalize = np.interp(self.velocity, (0, 100), (0, 1))
        velocity_dim = np.full((66, 200), velocity_normalize)
        velocity_tensor_image = np.dstack((tensor_img, velocity_dim))
        final_image = velocity_tensor_image[np.newaxis]
        
        # GradCAM from image
        cam = GradCAM(self.model, 0)
        heatmap = cam.compute_heatmap(final_image)
        heatmap = cv2.resize(heatmap, (heatmap.shape[0], heatmap.shape[1]))
        (heatmap, output) = cam.overlay_heatmap(heatmap, tensor_img, alpha=0.5)
        output = np.swapaxes(output, 0, 1)
        
        steer_val = self.w
        throttle_val = self.v
        brake_val = self.b
        self.velocity = current_velocity
        
        example_result = self.model.predict(final_image, verbose=0)
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

        if self.velocity > 30:
            throttle_val = 0.0
            brake_val = 0.0

        self.v = throttle_val
        self.w = steer_val

        """boxes = self.tlc.detect_multi_object(img,score_threshold=0.2)
        if len(boxes) != 0:
            print("SEMAFORO")"""

        return steer_val, throttle_val, brake_val, cropped, output
    

    def run_step_pilotnet_star(self, img, current_velocity):
        """
        Execute one step of navigation.
        :return: control
        """
        cropped = img[200:-1,:] # 240
        
        """hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
        YELLOW_MIN = np.array([15, 80, 80],np.uint8)
        YELLOW_MAX = np.array([35, 255, 255],np.uint8)
        frame_threshed = cv2.inRange(hsv, YELLOW_MIN, YELLOW_MAX)
        tensor_img = 
        
        tensor_img[frame_threshed>0]=(255,0,0)"""
        # cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        tensor_img = cv2.resize(cropped, (200, 66))/255.0
        final_image = tensor_img[np.newaxis]
        
        # GradCAM from image
        """cam = GradCAM(self.model, 0)
        heatmap = cam.compute_heatmap(final_image)
        heatmap = cv2.resize(heatmap, (heatmap.shape[0], heatmap.shape[1]))
        (heatmap, output) = cam.overlay_heatmap(heatmap, tensor_img, alpha=0.5)
        output = np.swapaxes(output, 0, 1)"""
        
        steer_val = self.w
        throttle_val = self.v
        brake_val = self.b
        self.velocity = current_velocity
        
        example_result = self.model.predict(final_image, verbose=0)
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

        if self.velocity > 30:
            throttle_val = 0.0
            brake_val = 0.0

        self.v = throttle_val
        self.w = steer_val

        return steer_val, throttle_val, brake_val, cropped#, output



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
        tensor_img = cv2.resize(cropped, (200, 66), interpolation = cv2.INTER_AREA)/255.0
        final_image = tensor_img[np.newaxis]
        
        steer_val = self.w
        throttle_val = self.v
        # if data.frame_number % 10 == 0:
        example_result = self.model.predict(final_image)
        # Follow lane steering values
        steer_val = np.interp(example_result[0][1], (0, 1), (-1, 1))
        # steer_val = np.interp(example_result[0], (0, 1), (-1, 1))
        if self.first < 100:
            throttle_val = 0.2
            self.first += 1
        else:
            throttle_val = 0.3
        # print(f"THROTTLE: {throttle_val} - STEERING {steer_val}")

        self.v = throttle_val
        self.w = steer_val

        return steer_val, throttle_val, img