""" This module implements an agent that follows a line along a track"""
import numpy as np
import math

import cv2


YELLOW_MIN = np.array([15, 80, 80],np.uint8)
YELLOW_MAX = np.array([35, 255, 255],np.uint8)


def search_along_y(frame_threshed, target_y):
    left_lane_x = 0
    right_lane_x = frame_threshed.shape[1]
    for x in range(0, frame_threshed.shape[1]):
            if frame_threshed[target_y, x] == 255:
                right_lane_x = x
                break
    for x in reversed(range(0, frame_threshed.shape[1])):
        if frame_threshed[target_y, x] == 255:
            left_lane_x = x
            break

    return left_lane_x, right_lane_x



class FLAgent(object):
    """
    Base class to define agents in CARLA
    """

    def __init__(self, vehicle):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._vehicle.get_world().get_map()
        self._prev_center = 0
        self._prev_orientation = 0
        self.in_lane = False


    def run_step(self, img):
        """
        Execute one step of navigation.
        :return: control
        """

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        frame_threshed = cv2.inRange(hsv, YELLOW_MIN, YELLOW_MAX)

        left_x, right_x = search_along_y(frame_threshed, (img.shape[0] - 1)) # CHECKING WETHER WE ARE ON A CURVE
        low_center = (int(left_x + (right_x - left_x) / 2))
        bottom_distance = abs(low_center - img.shape[1]/2)

        if bottom_distance > 150: # CURVE
            target_y = int(img.shape[0] * 0.7)
        else:
            target_y = int(img.shape[0] * 0.6)

        left_lane_x, right_lane_x = search_along_y(frame_threshed, target_y)

        center = (int(left_lane_x + (right_lane_x - left_lane_x) / 2), target_y)

        if (left_lane_x == 0) and (right_lane_x == img.shape[1]): # CAN'T FIND THE LANE
            target_y = 246
            left_lane_x, right_lane_x = search_along_y(frame_threshed, target_y)
            center = (int(left_lane_x + (right_lane_x - left_lane_x) / 2), target_y)

        distance_centers = center[0] - self._prev_center

        if (distance_centers == 0) or (distance_centers == -1):
            self.in_lane = True

        if self.in_lane:
            if abs(distance_centers) > 2:
                self.in_lane = False
                center = (self._prev_center, target_y)
        else:
            self._prev_center = center[0]

        if type(center).__module__ != np.__name__:
            cv2.circle(img_rgb, center, 3, (0, 0, 255), -1)

        # GET ORIENTATION OF THE CAR FROM THE LANE
        orientation = center[0] - img.shape[1] / 2

        diff = orientation - self._prev_orientation
        self._prev_orientation = orientation
        steer = (0.009 * orientation + 0.035 * diff)

        return steer, 0.4, img_rgb