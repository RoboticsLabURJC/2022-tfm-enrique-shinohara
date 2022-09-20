#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

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
from collections import deque
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
        self.v = .0
        self.w = .0
        self.fps = deque(maxlen=60)
        

# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(data, obj, vehicle, model):
    step_start = time.time()

    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:,:,:3]
    img = img[:, :, ::-1]
    cropped = img[246:-1,:]

    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    YELLOW_MIN = np.array([15, 80, 80],np.uint8)
    YELLOW_MAX = np.array([35, 255, 255],np.uint8)
    frame_threshed = cv2.inRange(hsv, YELLOW_MIN, YELLOW_MAX)
    tensor_img = cropped
    tensor_img[frame_threshed>0]=(255,0,0)

    tensor_img = cv2.resize(tensor_img, (200, 66), interpolation = cv2.INTER_AREA)
    final_image = tensor_img[np.newaxis]
    
    steer_val = obj.w
    throttle_val = obj.v
    if data.frame_number % 10 == 0:
        example_result = model.predict(final_image)
        steer_val = example_result[0][0]*5
        throttle_val = example_result[0][1]*0.5
    
    obj.v = throttle_val
    obj.w = steer_val
    # print(obj.v, obj.w)

    frame_time = time.time() - step_start
    obj.fps.append(frame_time)
    print(f'Agent: {len(obj.fps)/sum(obj.fps):>4.1f} FPS | {frame_time*1000} ms')

    vehicle.apply_control(carla.VehicleControl(throttle=float(obj.v), steer=float(obj.w)))

    # print(vehicle.get_location())

    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0,1))



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

    savedModel.predict(np.ones((1, 66, 200, 3)))

    # In this tutorial script, we are going to add a vehicle to the simulation
    # and let it drive in autopilot. We will also create a camera attached to
    # that vehicle, and save all the images generated by the camera to disk.

    try:
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

        camera.listen(lambda image: pygame_callback(image, renderObject, vehicle, savedModel))
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
        while (time.time() < t_end):
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
