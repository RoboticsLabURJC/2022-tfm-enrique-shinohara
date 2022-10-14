#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" Module with auxiliary functions. """

import math

import numpy as np
from cv2 import cv2


def check_leftright_lanes(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(img, 100, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (21, 2))

    dilated = cv2.dilate(edges, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=3)

    target_x = int(img.shape[1]/2)
    target_y = int(img.shape[0] * 0.64)

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
