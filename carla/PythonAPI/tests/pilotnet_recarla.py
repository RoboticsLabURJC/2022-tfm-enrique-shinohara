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
import argparse
import datetime
import collections
import cv2
import csv
from PIL import Image
from csv import writer
import tensorflow as tf
from collections import deque
from agents.rl_agent import *
from agents.vision_agent import *
from agents.followline_agent import *

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

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 10: Simply a right turn
# 17: Right turn with an annoying barrier which hitbox hits the car
SPAWNPOINT = 10


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../*%d.%d-%s.egg' % (
        sys.version_info.major,
        5,
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
        cam_pos_index = self.camera_manager._transform_index if self.camera_manager is not None else 0

        blueprint = self.world.get_blueprint_library().find('vehicle.lincoln.mkz2017')
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
            # spawn_point.rotation.yaw += 20
            # spawn_point.rotation.yaw -= 20
            # spawn_point.rotation.yaw += 15
            # spawn_point.rotation.yaw -= 10
            self.vehicle = self.world.spawn_actor(blueprint, spawn_point)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        self.camera_manager = CameraManager(self.vehicle, self.hud)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
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
        # self.hud.render(display)

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
        if True:
            return
        t = world.vehicle.get_transform()
        v = world.vehicle.get_velocity()
        c = world.vehicle.get_vehicle_control()
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
            'Server:  % 16d FPS' % self.server_fps,
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.vehicle, truncate=20),
            'Map:     % 20s' % world.world.map_name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'Height:  % 18.0f m' % t.location.z,
            '',
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            ('Reverse:', c.reverse),
            ('Hand brake:', c.hand_brake),
            ('Manual:', c.manual_gear_shift),
            'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear),
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
            """for item in self._info_text:
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
                if item: # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18"""
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
        self._surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=2.5, z=0.8))]
        self._transform_index = 1
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
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
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = 2

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        self._hud.notification('Recording %s' % ('On' if self._recording else 'Off'))

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, data):
        self = weak_self()
        if not self:
            return
        
        img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
        img = img[:,:,:3]
        img = img[:, :, ::-1]

        self.image = img
        # print(self._parent.get_velocity())

        # self._surface = pygame.surfarray.make_surface(croped_img.swapaxes(0, 1))


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def line_prepender(filename, line):
    with open(filename, "r") as infile:
        reader = list(csv.reader(infile))
        reader.insert(0, line)

    with open(filename, "w") as outfile:
        writer = csv.writer(outfile)
        for line in reader:
            writer.writerow(line)

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

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)

        if args.agent == "rl":
            agent = RLAgent(world.vehicle)
            agent.model.summary()
            agent.model.predict(np.ones((1, 66, 200, 3)))
        elif args.agent == "fl":
            agent = FLAgent(world.vehicle)
        else:
            agent = Agent(world.vehicle)


        if args.rec == True:
            if not os.path.exists('./_%d/' % SPAWNPOINT):
                os.makedirs('./_%d/' % SPAWNPOINT)


        clock = pygame.time.Clock()
        fps = deque(maxlen=60)
        # media = deque(maxlen=1000)
        image_counter = 0
        first_image_taken = 0
        recording = False
        while True:
            # as soon as the server is ready continue!
            if not world.world.wait_for_tick(10.0):
                continue

            step_start = time.time()
            if type(world.camera_manager.image).__module__ == np.__name__:
                steer, throttle, image = agent.run_step(world.camera_manager.image)
                world.camera_manager._surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
                world.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer)))

                image_counter = image_counter + 1

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit()

                    elif event.type == pygame.KEYDOWN:
                        if event.key == K_r and recording == False:
                            recording = True
                        elif event.key == K_r and recording == True:
                            recording = False

                # if (args.rec == True) and (image_counter % 10 == 0):
                if recording:
                    file_path = "_%d/" % (SPAWNPOINT)
                    file_name = "%d.png" % (image_counter) # / 10)
                    j = Image.fromarray(world.camera_manager.image)
                    j.save(file_path + file_name, compress_level=1)

                    row_contents = [file_name, throttle, steer]
                    # Append a list as new line to an old csv file
                    append_list_as_row(file_path + 'data.csv', row_contents)

                    if first_image_taken == 0:
                        line_prepender(file_path + 'data.csv', ['image', 'throttle', 'steer'])
                        first_image_taken = 1
            
            frame_time = time.time() - step_start
            fps.append(frame_time)

            #media.append(hud.server_fps)
            # if len(media) == 1000:
            #     print(f"MEDIA DE FPS DE SERVIDOR -----------------------------------------> {sum(media) / len(media)}")
            
            world.tick(clock)
            if args.debug == True:
                print(f"Server: {hud.server_fps}")
                print(f"Client: {len(fps)/sum(fps):>4.1f} FPS | {frame_time*1000} ms")

            world.render(display)
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

    argparser.add_argument("-r", "--record",
                        action='store_true',
                        dest='rec')

    argparser.add_argument("-a", "--agent", type=str,
                           choices=["rl", "basic", "fl"],
                           help="select which agent to run",
                           default="basic")
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

    main()
