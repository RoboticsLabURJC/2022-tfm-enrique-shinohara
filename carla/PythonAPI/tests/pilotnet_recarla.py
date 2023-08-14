#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
    Example of automatic vehicle control from client side.
"""

from __future__ import print_function

import os
import re
import sys
import math
import glob
import time
import pandas
import random
import logging
import weakref
import schedule
import argparse
import datetime
import threading
import collections
import cv2
import csv
from csv import writer
import tensorflow as tf
from collections import deque
from agents.rl_agent import *
from agents.vision_agent import *
from agents.followline_agent import *
import PIL

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# TOWN 02:
# 12 - left turn at the gas station
# 20 - right turn at the gas station


# 17 - left turn at the gas station
# 22 - right turn at the gas station
global SPAWNPOINT
global NPC_SPAWNPOINT
global NPC_NAME
SPAWNPOINT = 41
NPC_SPAWNPOINT = 43
# SPAWNPOINT = 28
# NPC_SPAWNPOINT = 5
# SPAWNPOINT = 73
# NPC_SPAWNPOINT = 148
# SPAWNPOINT = 121
# NPC_SPAWNPOINT = 207
NPC_NAME = 'vehicle.mini.cooper_s'
LEFT = False
RIGHT = False
CHECK07 = -1

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/*%d.%d-%s.egg' % (
        sys.version_info.major,
        7,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc



# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate-1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.hud = hud
        self.vehicle = None
        self.npc_vehicle = None
        self.collision_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.args = args

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager._index if self.camera_manager is not None else 0
        # cam_index = 5
        cam_pos_index = self.camera_manager._transform_index if self.camera_manager is not None else 0

        blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the vehicle.
        if self.vehicle is not None:
            spawn_point = self.vehicle.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()

            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = spawn_points[SPAWNPOINT]
            self.vehicle = self.world.spawn_actor(blueprint, spawn_point)

        while self.vehicle is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = spawn_points[SPAWNPOINT]
            spawn_point.location.x += 3
            random_starting_point = bool(random.getrandbits(1))
            """if random_starting_point:
                random_value = random.randint(20, 50)
                spawn_point.rotation.yaw += random_value
                print(f"RIGHT: {random_value}")
            else:
                random_value = random.randint(20, 50)
                spawn_point.rotation.yaw -= random_value
                print(f"LEFT: {random_value}")"""
                
            self.vehicle = self.world.spawn_actor(blueprint, spawn_point)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        self.camera_manager = CameraManager(self.vehicle, self.hud)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        # self.camera_manager.set_sensor_2(500, notify=False)
        actor_type = get_actor_display_name(self.vehicle)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.vehicle.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- HUD -----------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        t = world.vehicle.get_transform()
        v = world.vehicle.get_velocity()
        c = world.vehicle.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Vehicle: % 20s' % get_actor_display_name(world.vehicle, truncate=20),
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            '',
            'Throttle: %16.2f' % c.throttle,
            ('Throttle:', c.throttle, 0.0, 1.0),
            'Steer: %19.2f' % c.steer,
            ('Steer:', c.steer, -1.0, 1.0),
            'Brake: %19.2f' % c.brake,
            ('Brake:', c.brake, 0.0, 1.0),
            'Brake: %19.2f' % c.brake,
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)
        ]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.vehicle.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
        
        self._notifications.tick(world, clock)

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self._hud.notification('Collision with %r, id = %d' % (actor_type, event.other_actor.id))
        print(f"HISTORIA DE COLISIONES ---> {len(self._history)}")
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self._history.append((event.frame_number, intensity))
        if len(self._history) > 4000:
            self._history.pop(0)


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.sensor2 = None
        self._surface = None
        self._second_surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=2.5, z=0.8)),
            carla.Transform(carla.Location(x=-6, z=3))]
        self._transform_index = 1
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            item.append(bp)
        self._index = None
        self.image = None
        self.image2 = None

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self._transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            print(self.sensor)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def set_sensor_2(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        self.sensor2 = self._parent.get_world().spawn_actor(
            self._sensors[0][-1],
            self._camera_transforms[1],
            attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
        weak_self = weakref.ref(self)
        self.sensor2.listen(lambda image: CameraManager._parse_image_2(weak_self, image))
        self._index = 3

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        self._hud.notification('Recording %s' % ('On' if self._recording else 'Off'))

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))
            # display.blit(self._second_surface, (0, self._hud.dim[1]))

    @staticmethod
    def _parse_image(weak_self, data):
        self = weak_self()
        if not self:
            return

        """img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
        img = img[:,:,:3]
        img = img[:, :, ::-1]

        # j = Image.fromarray(img)
        # j.save("output.png")

        self.image = img"""

        data.convert(self._sensors[self._index][1])
        # array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(data.raw_data, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        self.image = array

    @staticmethod
    def _parse_image_2(weak_self, data):
        self = weak_self()
        if not self:
            return
        
        img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
        img = img[:,:,:3]
        img = img[:, :, ::-1]

        self.image2 = img


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def line_prepender(filename, line):
    """with open(filename, "r") as infile:
        reader = list(csv.reader(infile))
        reader.insert(0, line)

    with open(filename, "w") as outfile:
        writer = csv.writer(outfile)
        for line in reader:
            writer.writerow(line)"""
    with open(filename, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(line)


def read_last_line(filename, column):
    f1 = open(filename, "r")
    last_line = f1.readlines()[-1]
    f1.close()
    return last_line.split(',')[column]


def save_instance_to_dataset(world, image, image_counter, first_image_taken, prev_image_num, prev_velocity):
    file_path = "_%d/" % (SPAWNPOINT)
    if (first_image_taken == 0) and (not os.path.exists('./_%d/data.csv' % SPAWNPOINT)):
        line_prepender(file_path + 'data.csv', ['image', 'throttle', 'steer', 'brake', 'prevelocity'])
        first_image_taken = 1
        prev_velocity = 0
    if (first_image_taken == 0) and (os.path.exists('./_%d/data.csv' % SPAWNPOINT)):
        files_list = glob.glob(file_path + '*.png')
        files_list.sort(key=natural_keys)
        first_image_taken = 1
        prev_image_num = int(files_list[-1].split('/')[1].split('.')[0])
        prev_velocity = 0

    file_name = "%d.png" % (image_counter + prev_image_num)
    j = PIL.Image.fromarray(image)
    j.save(file_path + file_name, compress_level=1)

    c = world.vehicle.get_control()
    v = world.vehicle.get_velocity()
    row_contents = [file_name, c.throttle, c.steer, c.brake, prev_velocity]

    prev_velocity = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    # Append a list as new line to an old csv file
    append_list_as_row(file_path + 'data.csv', row_contents)

    return first_image_taken, prev_image_num, prev_velocity


def try_spawn_random_vehicles(world, client, num, spawnpoint, npc_name):
    actor_list = []
    
    spawn_points = world.world.get_map().get_spawn_points()

    if spawnpoint == None:
        random_waypoints = []
        while True:
            random_waypoints = random.choices(spawn_points, k=num)
            if spawn_points[SPAWNPOINT] in random_waypoints:
                continue
            else:
                break
        print(len(random_waypoints))
        for spawn_point in random_waypoints:
            blueprints = world.world.get_blueprint_library().filter('vehicle.*')
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
        
            vehicle = world.world.try_spawn_actor(blueprint, spawn_point)
            if vehicle is not None:
                actor_list.append(vehicle)
                vehicle.set_autopilot()
                print('spawned %r at %s' % (vehicle.type_id, spawn_point.location))
                world.npc_vehicle = vehicle
            else:
                return False
            time.sleep(0.5)
        return True
    else:
        blueprint = world.world.get_blueprint_library().find(npc_name)
        spawn_point = spawn_points[spawnpoint]
        vehicle = world.world.try_spawn_actor(blueprint, spawn_point)

        traffic_manager = client.get_trafficmanager()
        route = ["Straight"]*1000
        traffic_manager.set_route(vehicle, route)
        traffic_manager.ignore_lights_percentage(vehicle, 0)

        if vehicle is not None:
                actor_list.append(vehicle)
                vehicle.set_autopilot()
                print('spawned %r at %s' % (vehicle.type_id, spawn_point.location))
                world.npc_vehicle = vehicle
        else:
            return False
        return True


# ==============================================================================
# -- game_loop() ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)
        traffic_manager = client.get_trafficmanager()

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.load_world('Town02_Opt'), hud, args)
        world.world.unload_map_layer(carla.MapLayer.Particles)
        # world.world.unload_map_layer(carla.MapLayer.Buildings)
        # try_spawn_random_vehicles(world, client, 0, NPC_SPAWNPOINT, NPC_NAME)

        weather = world.world.get_weather()
        weather.wind_intensity = 00.0
        weather.cloudiness = 0.0#60.0
        weather.precipitation = 00.0
        weather.wind_intensity = 00.0
        weather.sun_azimuth_angle= 275.0
        weather.precipitation_deposits = 0.0#50
        weather.sun_altitude_angle=20#60.0#15.0
        weather.fog_density=5.0
        weather.fog_distance=0.75
        weather.fog_falloff=0.1
        weather.wetness=00.0
        world.world.set_weather(weather)
        """vehicle_light_state = carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
        world.vehicle.set_light_state(vehicle_light_state)"""

        model = 2 # (0: PilotNet* fl dataset | 1: PilotNet* 1 npc | 2: PilotNet** 1/x npc)

        if args.agent == "rl":
            agent = RLAgent(world.vehicle)
            agent.model.summary()
            if model == 2:
                agent.model.predict(np.ones((1, 66, 200, 4)))
            else:
                agent.model.predict(np.ones((1, 66, 200, 3)))
        elif args.agent == "fl":
            agent = FLAgent(world.vehicle)
        elif args.agent == "fr":
            agent = Agent(world.vehicle)
        else:
            agent = None
            world.vehicle.set_autopilot(True)

            route = ["Straight"]*1000
            # route = ["Right", "Straight", "Straight", "Straight", "Straight", "Straight", "Right"]*500 # Town 07 left WAYPOINT: 57
            # route = ["Right", "Right", "Straight", "Straight", "Straight", "Straight", "Straight"]*500 # Town 07 left WAYPOINT: 70
            # route = ["Straight", "Straight", "Straight", "Straight", "Right", "Right", "Straight"]*500 # Town 07 left WAYPOINT: 97
            # if CHECK07 == 0:
            #     route = ["Straight", "Left", "Left", "Straight", "Straight", "Straight", "Straight"]*500 # Town 07 right WAYPOINT: 24
            # elif CHECK07 == 1:
            #     route = ["Straight", "Straight", "Straight", "Straight", "Straight", "Left", "Left"]*500 # Town 07 right WAYPOINT: 46
            # elif CHECK07 == 2:
            #     route = ["Straight", "Straight", "Straight", "Straight", "Left", "Left", "Straight"]*500 # Town 07 right WAYPOINT: 71
            traffic_manager.set_route(world.vehicle, route)
            traffic_manager.ignore_lights_percentage(world.vehicle, 100)
            traffic_manager.ignore_signs_percentage(world.vehicle, 100)
            traffic_manager.distance_to_leading_vehicle(world.vehicle, 15)
            traffic_manager.auto_lane_change(world.vehicle, False)
            # traffic_manager.keep_right_rule_percentage(world.vehicle, 100)
            
        # traffic_manager.vehicle_percentage_speed_difference(world.npc_vehicle, 50)

        if args.rec != "":
            if not os.path.exists('./_%d/' % SPAWNPOINT):
                os.makedirs('./_%d/' % SPAWNPOINT)


        clock = pygame.time.Clock()
        fps = deque(maxlen=60)
        media = deque(maxlen=1000)
        image_counter = 0
        first_image_taken = 0
        recording = False
        noise = False
        noise_value = 0
        prev_image_number = 0
        prev_velocity = 0
        stop_condition = False
        stop_counter = 0
        start_record = False
        start_counter = 0
        pre_steer = 0
        noise_counter = 0
        start_location = None
        route = ["Straight"]*1000
        while True:
            # as soon as the server is ready continue!
            if not world.world.wait_for_tick(10.0):
                continue
            step_start = time.time()
            control = world.vehicle.get_control()
            location = world.vehicle.get_location()
            if image_counter == 0:
                start_location = location

            if image_counter > 50:
                if ((location.x > start_location.x - 1) and (location.x <= start_location.x + 1)) and ((location.y > start_location.y - 1) and (location.y <= start_location.y + 1)):
                    print("VUELTA COMPLETADA!!!!")
                    break

            
            # npc_control = world.npc_vehicle.get_control()
            # npc_transform = world.npc_vehicle.get_transform()
            if (type(world.camera_manager.image).__module__ == np.__name__):
                # If we use autopilot of carla and no agent to control the car
                if agent == None:
                    world.camera_manager._surface = pygame.surfarray.make_surface(world.camera_manager.image.swapaxes(0, 1))

                    if world.vehicle.is_at_traffic_light():
                        traffic_light = world.vehicle.get_traffic_light()
                        traffic_light.set_red_time(5)

                    image_counter = image_counter + 1
                # The agent selected will be used to control the car
                else:
                    world.camera_manager._surface = pygame.surfarray.make_surface(world.camera_manager.image.swapaxes(0, 1))

                    if model == 2:
                        v = world.vehicle.get_velocity()
                        current_velocity = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                        steer, throttle, brake, image, network_image = agent.run_step_pilotnet_star_star(world.camera_manager.image, current_velocity)
                        # steer, throttle, brake, image = agent.run_step_pilotnet_star_star(world.camera_manager.image, current_velocity)
                        # world.camera_manager._second_surface = pygame.surfarray.make_surface(world.camera_manager.image2.swapaxes(0, 1))
                        world.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake)))
                        image_counter = image_counter + 1
                    if model == 1:
                        v = world.vehicle.get_velocity()
                        current_velocity = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                        # NO PRE VELOCITY 200x66x3
                        steer, throttle, brake, image = agent.run_step_pilotnet_star(world.camera_manager.image, current_velocity)
                        world.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake)))
                        image_counter = image_counter + 1
                    if model == 0:
                        steer, throttle, image = agent.run_step_old(world.camera_manager.image)
                        world.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer)))
                        """name = "zoutput_image_" + str(image_counter) + ".png"
                        name2 = "zoutput_network_image_" + str(image_counter) + ".png"
                        cv2.imwrite(name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(name2, np.swapaxes(network_image, 0, 1))"""
                        image_counter = image_counter + 1

                """if world.npc_vehicle != None:
                    # Stop the spawned car at random times
                    random_value = random.random()
                    if (random_value < 0.005) and (stop_condition == False) and ((abs(npc_control.steer) <=0.1) or (abs(control.steer) <= 0.1)):
                        print("Stopping....")
                        stop_condition = True
                    if stop_condition:
                        if stop_counter < 250:
                            npc_control.throttle = 0
                            npc_control.brake = 1
                            world.npc_vehicle.apply_control(npc_control)
                            stop_counter += 1
                        else:
                            print("Run!")
                            stop_condition = False
                            stop_counter = 0
                            npc_control.throttle = 0
                            npc_control.brake = 0
                            world.npc_vehicle.apply_control(npc_control)"""

                """if (noise == False) and (image_counter % 200 == 0):
                    noise = True
                elif (noise == True) and (image_counter % 15 == 0):
                    noise = False
                    noise_value = 0

                # Apply noise if activated
                if noise:
                    noise_value = random.uniform(-0.03, 0.03)
                    control.steer += noise_value
                    world.vehicle.apply_control(control)"""

                """for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit()

                    elif event.type == pygame.KEYDOWN:
                        if event.key == K_r and recording == False:
                            recording = True
                        elif event.key == K_r and recording == True:
                            recording = False"""
                            
                """if ((139 < location.x < 163) and (-194 < location.y < 193.5)) and RIGHT: # Map03 right turn
                    continue
                if (134 < location.x < 170) and (-207.89 < location.y < -204) and LEFT: # Map03 left turn
                    continue"""
                """if ((70 < location.x < 75.4) and (-17 < location.y < 8.5)) or ((68 < location.x < 77) and (47 < location.y < 75)) and RIGHT: # Map07 right turn
                    continue
                if ((68 < location.x < 74) and (-27 < location.y < 7)) or ((68 < location.x < 78) and (47 < location.y < 68)) and LEFT: # Map07 left turn
                    continue"""
                
                # TURNING RECORDING
                """if abs(control.steer) >= 0.09: # 0.3
                    print(f"GIROOOOOOO - {image_counter}")
                    recording = True
                else:
                    recording = False"""

                # ACCELERATING RECORDING
                """velocity = world.vehicle.get_velocity()
                my_velocity = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                if abs(my_velocity) >= 35:
                    print(f"ACELERANDO - {image_counter}")
                    recording = True
                else:
                    recording = False"""
                
                # START THE ENGINE
                """velocity = world.vehicle.get_velocity()
                my_velocity = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                if my_velocity <= 0.01:
                    # Car has stopped, reset recording status
                    start_counter += 1
                    if start_counter > 80:
                        recording = False
                    start_record = True
                if start_record and 1 < my_velocity < 20:
                    # Car has started accelerating, start recording
                    recording = True
                if recording and my_velocity > 25:
                    # Car has exceeded recording speed, stop recording
                    recording = False
                    start_counter = 0
                    start_record = True"""
                # Apply noise if activated
                """if (1 < my_velocity < 25) and (noise_counter < 100):
                    noise_counter += 1
                    noise_value = random.uniform(-0.1, 0.1)
                    print(abs((control.steer + noise_value) - pre_steer))
                    if abs((control.steer + noise_value) - pre_steer) < 0.4:
                        control.steer += noise_value
                    else:
                        control.steer += -1*noise_value
                    world.vehicle.apply_control(control)
                elif ((noise_counter > 100) or (my_velocity > 25)) and noise_counter != 0:
                    noise_counter = 0
                    pre_steer = control.steer"""


                # if (args.rec == True) and (image_counter % 10 == 0):
                if args.rec == "rgb":
                    print(f"GRABANDOOOOOOOO - {image_counter}")
                    first_image_taken, prev_image_number, prev_velocity = save_instance_to_dataset(world, world.camera_manager.image, 
                    image_counter, first_image_taken, prev_image_number, prev_velocity)

                    """if len(glob.glob("_%d/" % (SPAWNPOINT) + '*.png')) > 4:
                        break"""

                    if image_counter > 10:
                        break
            
            frame_time = time.time() - step_start
            fps.append(frame_time)

            """media.append(hud.server_fps)
            if len(media) == 1000:
                print(f"MEDIA DE FPS DE SERVIDOR -----------------------------------------> {sum(media) / len(media)}")"""
            
            world.tick(clock)
            if args.debug == True:
                print(f"Server: {hud.server_fps}")
                print(f"Client: {len(fps)/sum(fps):>4.1f} FPS | {frame_time*1000} ms")


            world.render(display)
            # display.blit(pygame.surfarray.make_surface(network_image), (args.width - 200, 0))
            pygame.display.flip()


    finally:
        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='640x480',
        help='window resolution (default: 640x480)')

    argparser.add_argument("-r", "--rec",
                           choices=["rgb", "bird"],
                           help="select which view to record",
                           default="")

    argparser.add_argument("-a", "--agent", type=str,
                           choices=["rl", "basic", "fl", "fr"],
                           help="select which agent to run",
                           default="basic")

    argparser.add_argument("-c", "--choice", type=int,
                           help="select which main option to run",
                           default=3)
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)



if __name__ == '__main__':
    choice = int(sys.argv[2])
    if choice == 0:  # Run clockwise and anticlockwise circuits with deletable traces
        start = time.time()
        spawnpoints_left = []
        spawnpoints_right = [70]
        LEFT = True
        for spawn in spawnpoints_left:
            SPAWNPOINT = spawn
            main()
        LEFT = False
        RIGHT = True
        for index, spawn in enumerate(spawnpoints_right):
            CHECK07 = index
            print(CHECK07)
            SPAWNPOINT = spawn
            main()
        RIGHT = False
        end = time.time() - start
        print(f"THE END: time -> {end}")
    elif choice == 1:  # Run circuits with no deletable traces
        start = time.time()
        spawnpoints = [41, 94]
        for spawn in spawnpoints:
            SPAWNPOINT = spawn
            main()
        end = time.time() - start
        print(f"THE END: time -> {end}")
    elif choice == 2:  # Run circuits with a variety of npcs
        start = time.time()
        spawnpoints = [40]*3
        npc_spawnpoints = [92]*3
        npc_names = ['vehicle.chevrolet.impala', 'vehicle.ford.crown', 'vehicle.tesla.cybertruck']
        for spawn, npcs, names in zip(spawnpoints, npc_spawnpoints, npc_names):
            SPAWNPOINT = spawn
            NPC_SPAWNPOINT = npcs
            NPC_NAME = names
            main()
        end = time.time() - start
        print(f"THE END: time -> {end}")
    elif choice == 3:  # A simple code
        main()
    else:
        print("Invalid choice. Please enter a number between 0 and 3.")