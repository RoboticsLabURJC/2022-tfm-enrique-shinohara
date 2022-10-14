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

import cv2


def check_bottomtop_lanes(img):
    edges = cv2.Canny(img, 100, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (21, 2))

    dilated = cv2.dilate(edges, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=3)

    target_x = int(img.shape[1]/2)

    distance = 0
    possible_curve = 0
    for y in reversed(range(0, img.shape[0])):
        distance += 1
        if eroded[y, target_x] == 255:
            possible_curve = y
            break

    return distance, possible_curve


def check_leftright_lanes(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges = cv2.Canny(img, 100, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (21, 2))

    dilated = cv2.dilate(edges, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=3)

    target_x = int(img.shape[1]/2)
    target_y = int(img.shape[0] * 0.67) # 0.64

    left_lane_x = 0
    right_lane_x = img.shape[1]
    for x in range(target_x, img.shape[1]):
        if eroded[target_y, x] == 255:
            right_lane_x = x
            break
    for x in reversed(range(0, target_x)):
        if eroded[target_y, x] == 255:
            left_lane_x = x
            break

    return (left_lane_x, target_y), (right_lane_x, target_y), (target_x, target_y), img_rgb

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360
    
    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 
        
        return ang_deg


class Agent(object):
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
        self._prev_steer = 0
        self._road_distance = 400


    def run_step(self, img):
        """
        Execute one step of navigation.
        :return: control
        """

        left, right, center, croped_img = check_leftright_lanes(img)
        distance_curve, point_to_curve = check_bottomtop_lanes(img)
        # distance_left = np.linalg.norm(np.array(left) - np.array(center))
        # distance_right = np.linalg.norm(np.array(right) - np.array(center))

        if type(left).__module__ != np.__name__:
            cv2.circle(croped_img, left, 3, (0, 0, 255), -1)
        if type(right).__module__ != np.__name__:
            cv2.circle(croped_img, right, 3, (0, 0, 255), -1)

        left_v = np.subtract((int(img.shape[1]/2), int(img.shape[0])), left)
        right_v = np.subtract((int(img.shape[1]/2), int(img.shape[0])), right)
        direction = np.add(left_v, right_v)
        orientation = np.subtract((int(img.shape[1]/2), int(img.shape[0])), direction)

        cv2.line(croped_img, (int(img.shape[1]/2), int(img.shape[0])), (int(orientation[0]), int(orientation[1])), (53, 190, 33), 2)
        cv2.line(croped_img, (int(img.shape[1]/2), int(img.shape[0])),(int(img.shape[1]/2), 0), (255, 0, 0), 2)

        # CONTROLLER OF THE VEHICLE
        line1 = ((int(img.shape[1]/2), int(img.shape[0])),(int(img.shape[1]/2), 0))
        line2 = (int(img.shape[1]/2), int(img.shape[0])),(int(orientation[0]), int(orientation[1]))
        angle = 0
        steer = 0
        distance_road = right[0] - left[0]

        if distance_road > 500:  # Straight line without left lane
            steer = 0
        elif point_to_curve >= 261:  # Probably starting a curve
            distance_left = np.linalg.norm(np.array((int(img.shape[1]/2), int(img.shape[0]))) - np.array(left))
            distance_right = np.linalg.norm(np.array((int(img.shape[1]/2), int(img.shape[0]))) - np.array(right))
            if orientation[0] < int(img.shape[1]/2):  # Left section
                if distance_right <= 200:
                    diff = orientation[0] - self._prev_steer
                    steer = (0.001 * (orientation[0] - int(img.shape[1]/2)) + 0.0004 * diff)
            elif orientation[0] > int(img.shape[1]/2):  # Right section
                if distance_left <= 200:
                    diff = orientation[0] - self._prev_steer
                    steer = (0.0006 * (orientation[0] - int(img.shape[1]/2)) + 0.0004 * diff)  
        else:
            if orientation[0] < int(img.shape[1]/2):  # Left section
                angle = -1*ang(line1, line2)
            elif orientation[0] > int(img.shape[1]/2):  # Right section
                angle = ang(line1, line2)
            steer += 0.02 * angle
            
        self._prev_steer = steer
        throttle = 0.4

        return steer, throttle, croped_img