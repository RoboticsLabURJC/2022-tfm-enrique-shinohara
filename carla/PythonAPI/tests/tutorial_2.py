#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

"""try:
    sys.path.append(glob.glob('../*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass"""

try:
    sys.path.append(glob.glob('../*%d.%d-%s.egg' % (
        sys.version_info.major,
        5,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import math
import time
import pygame
import random
import cv2 as cv2 
import numpy as np
import open3d as o3d
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


IM_WIDTH = 640
IM_HEIGHT = 480
COLLISION_LIST = []


# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0,255,(height,width,3),dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))
        

# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(data, obj, vehicle):
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:,:,:3]
    img = img[:, :, ::-1]

    left, right, center, croped_img = check_leftright_lanes(img)
    # distance_left = np.linalg.norm(np.array(left) - np.array(center))
    # distance_right = np.linalg.norm(np.array(right) - np.array(center))

    if type(left).__module__ != np.__name__:
        cv2.circle(croped_img, left, 3, (0, 0, 255), -1)
    if type(right).__module__ != np.__name__:
        cv2.circle(croped_img, right, 3, (0, 0, 255), -1)

    left_v = np.subtract((int(data.width/2), int(data.height)), left)
    right_v = np.subtract((int(data.width/2), int(data.height)), right)
    direction = np.add(left_v, right_v)
    orientation = np.subtract((int(data.width/2), int(data.height)), direction)
    cv2.line(croped_img, (int(data.width/2), int(data.height)), (int(orientation[0]), int(orientation[1])), (53, 190, 33), 2)
    cv2.line(croped_img, (int(data.width/2), int(data.height)),(int(data.width/2), 0), (255, 0, 0), 2)

    # CONTROLLER OF THE VEHICLE
    line1 = ((int(data.width/2), int(data.height)),(int(data.width/2), 0))
    line2 = (int(data.width/2), int(data.height)),(int(orientation[0]), int(orientation[1]))
    angle = 0
    steer = 0
    distance = np.linalg.norm(np.array(left) - np.array(right))
    if distance == data.width:
        steer += 0.015 * 40
    else:
        if left[0] == 0:
            steer += 0.0015 * 30
        """elif right[0] == 0:
            steer += 0.002 * -30
        else:"""
        if orientation[0] < int(img.shape[1]/2):  # Left section
            angle = -1*ang(line1, line2)
        elif orientation[0] > int(img.shape[1]/2):  # Right section
            angle = ang(line1, line2)
        steer += 0.02 * angle
    vehicle.apply_control(carla.VehicleControl(throttle=float(0.6), steer=steer))

    # print(vehicle.get_location())

    obj.surface = pygame.surfarray.make_surface(croped_img.swapaxes(0,1))


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


def collision_handler(event):
    COLLISION_LIST.append(event)


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


def save_image(image):
    number = image.frame_number
    image = np.array(image.raw_data)
    i2 = image.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    j = Image.fromarray(i3)
    j.save("_out/%06d.png" % number)


def lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with intensity
    colors ready to be consumed by Open3D"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 3), 3))

    # Isolate the 3D data
    points = data[:, :-1]

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points[:, :1] = -points[:, :1]

    # # An example of converting points from sensor to vehicle space if we had
    # # a carla.Transform variable named "tran":
    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    # points = np.dot(tran.get_matrix(), points.T).T
    # points = points[:, :-1]

    point_list.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("out_name.ply", point_list)


def main():
    actor_list = []
    collision_detect = False

    savedModel = load_model('20220405-115235_pilotnet_new_dataset_opencv_plus_difficult_cases_extreme_cases_support_100_epochs_new_checkpoint_monitor_cp.h5')
    savedModel.summary()

    im = cv2.imread('44.png')
    im = cv2.resize(im, (200, 66), interpolation = cv2.INTER_AREA)
    im = im[np.newaxis]

    # In this tutorial script, we are going to add a vehicle to the simulation
    # and let it drive in autopilot. We will also create a camera attached to
    # that vehicle, and save all the images generated by the camera to disk.

    try:
        print("hello1")
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        # Once we have a client we can retrieve the world that is currently
        # running.
        world = client.get_world()

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        bp = blueprint_library.filter('model3')[0]

        # A blueprint contains the list of attributes that define a vehicle
        # instance, we can read them and modify some of them. For instance,
        # let's randomize its color.
        color = random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color', color)

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        transform = random.choice(world.get_map().get_spawn_points())

        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(bp, transform)

        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(True)

        # Let's add now a "depth" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        # camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.8))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)

        # Now let's add the LIDAR to the vehicle
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', str(128))
        lidar_bp.set_attribute('rotation_frequency', '100')
        lidar_bp.set_attribute('points_per_second', str(120000))
        lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        actor_list.append(lidar)
        print('created %s' % lidar.type_id)

        """collision_sensor = world.spawn_actor(blueprint_library.find('sensor.other.collision'), camera_transform, attach_to=vehicle)
        actor_list.append(collision_sensor)
        print('created %s' % collision_sensor.type_id)"""
        
        # Now we register the function that will be called each time the sensor
        # receives an image. In this example we are saving the image to disk
        # camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame_number))

        if not os.path.exists('./_out/'):
            os.makedirs('./_out/')
        if not os.path.exists('./_lidar_output/'):
            os.makedirs('./_lidar_output/')

        renderObject = RenderObject(IM_WIDTH, IM_HEIGHT)
        point_list = o3d.geometry.PointCloud()

        camera.listen(lambda image: pygame_callback(image, renderObject, vehicle))
        # camera.listen(lambda image: save_image(image))
        # lidar.listen(lambda point_cloud: lidar_callback(point_cloud, point_list))
        # lidar.listen(lambda LidarMeasurement: LidarMeasurement.save_to_disk('_lidar_output/%06d.ply' % LidarMeasurement.frame_number))
        # collision_sensor.listen(lambda event: collision_handler(event))

        # Oh wait, I don't like the location we gave to the vehicle, I'm going
        # to move it a bit forward.
        location = vehicle.get_location()
        location.z += 2
        vehicle.set_location(location)
        print('moved vehicle to %s' % location)

        # Initialise the display
        pygame.init()
        gameDisplay = pygame.display.set_mode((IM_WIDTH, IM_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        # Draw black to the display
        gameDisplay.fill((0,0,0))
        gameDisplay.blit(renderObject.surface, (0,0))
        pygame.display.flip()

        t_end = time.time() + 60*5
        steer = 0
        while (time.time() < t_end):
            t1 = time.time()
            example_result = savedModel.predict(im)
            print(time.time() - t1)
            
            gameDisplay.blit(renderObject.surface, (0,0))
            pygame.display.flip()

        
        pygame.quit()

    finally:

        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')

if __name__ == '__main__':

    main()
