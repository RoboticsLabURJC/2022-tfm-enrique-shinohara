<h1>Python API Reference</h1>

!!! important
    Versions prior to 0.9.0 have a very different API. For the documentation of
    the stable version please switch to the
    [stable branch](https://carla.readthedocs.io/en/stable/).

## `carla.Client`

- `Client(host, port, worker_threads=0)`
- `set_timeout(float_seconds)`
- `get_client_version()`
- `get_server_version()`
- `get_world()`

## `carla.World`

- `id`
- `map_name`
- `debug`
- `get_blueprint_library()`
- `get_map()`
- `get_spectator()`
- `get_weather()`
- `set_weather(weather_parameters)`
- `get_actors()`
- `spawn_actor(blueprint, transform, attach_to=None)`
- `try_spawn_actor(blueprint, transform, attach_to=None)`
- `wait_for_tick(seconds=1.0)`
- `on_tick(callback)`

## `carla.DebugHelper`

- `draw_point(location, size=0.1, color=carla.Color(), life_time=-1.0, persistent_lines=True)`
- `draw_line(begin, end, thickness=0.1, color=carla.Color(), life_time=-1.0, persistent_lines=True)`
- `draw_arrow(begin, end, thickness=0.1, arrow_size=0.1, color=carla.Color(), life_time=-1.0, persistent_lines=True)`
- `draw_box(box, rotation, thickness=0.1, color=carla.Color(), life_time=-1.0, persistent_lines=True)`
- `draw_string(location, text, draw_shadow=False, color=carla.Color(), life_time=-1.0, persistent_lines=True)`

## `carla.BlueprintLibrary`

- `find(id)`
- `filter(wildcard_pattern)`
- `__getitem__(pos)`
- `__len__()`
- `__iter__()`

## `carla.ActorBlueprint`

- `id`
- `tags`
- `has_tag(tag)`
- `match_tags(wildcard_pattern)`
- `has_attribute(key)`
- `get_attribute(key)`
- `set_attribute(key, value)`
- `__len__()`
- `__iter__()`

## `carla.ActorAttribute`

- `id`
- `type`
- `recommended_values`
- `is_modifiable`
- `as_bool()`
- `as_int()`
- `as_float()`
- `as_str()`
- `as_color()`
- `__eq__(other)`
- `__ne__(other)`
- `__nonzero__()`
- `__bool__()`
- `__int__()`
- `__float__()`
- `__str__()`

## `carla.ActorList`

- `find(id)`
- `filter(wildcard_pattern)`
- `__getitem__(pos)`
- `__len__()`
- `__iter__()`

## `carla.Actor`

- `id`
- `type_id`
- `parent`
- `semantic_tags`
- `is_alive`
- `attributes`
- `get_world()`
- `get_location()`
- `get_transform()`
- `get_velocity()`
- `get_acceleration()`
- `set_location(location)`
- `set_transform(transform)`
- `set_simulate_physics(enabled=True)`
- `destroy()`

## `carla.Vehicle(carla.Actor)`

- `bounding_box`
- `apply_control(vehicle_control)`
- `get_vehicle_control()`
- `set_autopilot(enabled=True)`

## `carla.TrafficLight(carla.Actor)`

- `state`

## `carla.Sensor(carla.Actor)`

- `is_listening`
- `listen(callback_function)`
- `stop()`

## `carla.SensorData`

- `frame_number`
- `transform`

## `carla.Image(carla.SensorData)`

- `width`
- `height`
- `fov`
- `raw_data`
- `convert(color_converter)`
- `save_to_disk(path, color_converter=None)`
- `__len__()`
- `__iter__()`
- `__getitem__(pos)`
- `__setitem__(pos, color)`

## `carla.LidarMeasurement(carla.SensorData)`

- `horizontal_angle`
- `channels`
- `raw_data`
- `get_point_count(channel)`
- `save_to_disk(path)`
- `__len__()`
- `__iter__()`
- `__getitem__(pos)`
- `__setitem__(pos, location)`

## `carla.CollisionEvent(carla.SensorData)`

- `actor`
- `other_actor`
- `normal_impulse`

## `carla.LaneInvasionEvent(carla.SensorData)`

- `actor`
- `crossed_lane_markings`

## `carla.VehicleControl`

- `throttle`
- `steer`
- `brake`
- `hand_brake`
- `reverse`
- `__eq__(other)`
- `__ne__(other)`

## `carla.Map`

- `name`
- `get_spawn_points()`
- `get_waypoint(location, project_to_road=True)`
- `get_topology()`
- `generate_waypoints(distance)`
- `to_opendrive()`
- `save_to_disk(path=self.name)`

## `carla.Waypoint`

- `transform`
- `is_intersection`
- `lane_width`
- `road_id`
- `lane_id`
- `next(distance)`

## `carla.WeatherParameters`

- `cloudyness`
- `precipitation`
- `precipitation_deposits`
- `wind_intensity`
- `sun_azimuth_angle`
- `sun_altitude_angle`
- `__eq__(other)`
- `__ne__(other)`

Static presets

- `carla.WeatherParameters.ClearNoon`
- `carla.WeatherParameters.CloudyNoon`
- `carla.WeatherParameters.WetNoon`
- `carla.WeatherParameters.WetCloudyNoon`
- `carla.WeatherParameters.MidRainyNoon`
- `carla.WeatherParameters.HardRainNoon`
- `carla.WeatherParameters.SoftRainNoon`
- `carla.WeatherParameters.ClearSunset`
- `carla.WeatherParameters.CloudySunset`
- `carla.WeatherParameters.WetSunset`
- `carla.WeatherParameters.WetCloudySunset`
- `carla.WeatherParameters.MidRainSunset`
- `carla.WeatherParameters.HardRainSunset`
- `carla.WeatherParameters.SoftRainSunset`

## `carla.Vector3D`

- `x`
- `y`
- `z`
- `__add__(other)`
- `__sub__(other)`
- `__eq__(other)`
- `__ne__(other)`

## `carla.Location`

- `x`
- `y`
- `z`
- `distance(other)`
- `__add__(other)`
- `__sub__(other)`
- `__eq__(other)`
- `__ne__(other)`

## `carla.Rotation`

- `pitch`
- `yaw`
- `roll`
- `__eq__(other)`
- `__ne__(other)`

## `carla.Transform`

- `location`
- `rotation`
- `__eq__(other)`
- `__ne__(other)`

## `carla.BoundingBox`

- `location`
- `extent`
- `__eq__(other)`
- `__ne__(other)`

## `carla.Timestamp`

- `frame_count`
- `elapsed_seconds`
- `delta_seconds`
- `platform_timestamp`
- `__eq__(other)`
- `__ne__(other)`

## `carla.Color`

- `r`
- `g`
- `b`
- `a`
- `__eq__(other)`
- `__ne__(other)`

## `carla.ColorConverter`

- `Raw`
- `Depth`
- `LogarithmicDepth`
- `CityScapesPalette`

## `carla.ActorAttributeType`

- `Bool`
- `Int`
- `Float`
- `RGBColor`

## `carla.TrafficLightState`

- `Off`
- `Red`
- `Yellow`
- `Green`
- `Unknown`

## `carla.LaneMarking`

- `Other`
- `Broken`
- `Solid`
