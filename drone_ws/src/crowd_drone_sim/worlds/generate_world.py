"""
generate_world.py — builds crowd_world.world programmatically.

WHY: hand-writing a new <actor> block for every pedestrian doesn't scale.
This script generates them in a loop instead — to add more pedestrians
later, just change N_PEDESTRIANS below (or pass --n_pedestrians) and
re-run. Everything else (drone with 4 cameras, fairground objects) stays
the same every time.

Run:
    python3 generate_world.py --n_pedestrians 6
    python3 generate_world.py --n_pedestrians 20 --arena_half 6

This OVERWRITES crowd_world.world in the current directory by default
(use --output to change that).
"""

import argparse
import math
import random


# ---------------------------------------------------------------------
# Fixed parts of the world (drone + fairground objects) — unchanged
# regardless of pedestrian count.
# ---------------------------------------------------------------------

HEADER = """<?xml version="1.0"?>
<sdf version="1.6">
  <world name="crowd_world">
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros>
        <namespace>/gazebo</namespace>
      </ros>
    </plugin>

    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
"""

DRONE_BLOCK = """
    <!-- =========================================================
         DRONE — quadcopter-style body with 4 cameras (front/back/
         left/right), each a real Gazebo camera sensor.
         ========================================================= -->
    <model name="drone">
      <pose>0 0 4.5 0 0 0</pose>
      <link name="link">
        <gravity>0</gravity>
        <inertial>
          <mass>1.0</mass>
        </inertial>

        <collision name="collision">
          <geometry>
            <box><size>0.5 0.5 0.15</size></box>
          </geometry>
        </collision>

        <visual name="body">
          <geometry><box><size>0.35 0.35 0.12</size></box></geometry>
          <material><ambient>0.15 0.15 0.15 1</ambient><diffuse>0.15 0.15 0.15 1</diffuse></material>
        </visual>

        <visual name="arm_fr"><pose>0.28 0.28 0 0 0 0.785</pose>
          <geometry><cylinder><radius>0.02</radius><length>0.5</length></cylinder></geometry>
          <material><ambient>0.2 0.2 0.2 1</ambient></material></visual>
        <visual name="arm_fl"><pose>0.28 -0.28 0 0 0 -0.785</pose>
          <geometry><cylinder><radius>0.02</radius><length>0.5</length></cylinder></geometry>
          <material><ambient>0.2 0.2 0.2 1</ambient></material></visual>
        <visual name="arm_br"><pose>-0.28 0.28 0 0 0 -0.785</pose>
          <geometry><cylinder><radius>0.02</radius><length>0.5</length></cylinder></geometry>
          <material><ambient>0.2 0.2 0.2 1</ambient></material></visual>
        <visual name="arm_bl"><pose>-0.28 -0.28 0 0 0 0.785</pose>
          <geometry><cylinder><radius>0.02</radius><length>0.5</length></cylinder></geometry>
          <material><ambient>0.2 0.2 0.2 1</ambient></material></visual>

        <visual name="rotor_fr"><pose>0.45 0.45 0.03 0 0 0</pose>
          <geometry><cylinder><radius>0.12</radius><length>0.02</length></cylinder></geometry>
          <material><ambient>0.9 0.1 0.1 1</ambient></material></visual>
        <visual name="rotor_fl"><pose>0.45 -0.45 0.03 0 0 0</pose>
          <geometry><cylinder><radius>0.12</radius><length>0.02</length></cylinder></geometry>
          <material><ambient>0.1 0.3 0.9 1</ambient></material></visual>
        <visual name="rotor_br"><pose>-0.45 0.45 0.03 0 0 0</pose>
          <geometry><cylinder><radius>0.12</radius><length>0.02</length></cylinder></geometry>
          <material><ambient>0.1 0.3 0.9 1</ambient></material></visual>
        <visual name="rotor_bl"><pose>-0.45 -0.45 0.03 0 0 0</pose>
          <geometry><cylinder><radius>0.12</radius><length>0.02</length></cylinder></geometry>
          <material><ambient>0.9 0.1 0.1 1</ambient></material></visual>

        <sensor name="camera_front" type="camera">
          <pose>0.18 0 0 0 0 0</pose>
          <camera><horizontal_fov>1.4</horizontal_fov><image><width>320</width><height>240</height></image><clip><near>0.1</near><far>100</far></clip></camera>
          <always_on>true</always_on><update_rate>10</update_rate><visualize>false</visualize>
          <plugin name="cam_front_plugin" filename="libgazebo_ros_camera.so">
            <ros><namespace>/drone/camera_front</namespace></ros>
            <camera_name>front</camera_name><frame_name>camera_front_link</frame_name>
          </plugin>
        </sensor>
        <sensor name="camera_back" type="camera">
          <pose>-0.18 0 0 0 0 3.14159</pose>
          <camera><horizontal_fov>1.4</horizontal_fov><image><width>320</width><height>240</height></image><clip><near>0.1</near><far>100</far></clip></camera>
          <always_on>true</always_on><update_rate>10</update_rate><visualize>false</visualize>
          <plugin name="cam_back_plugin" filename="libgazebo_ros_camera.so">
            <ros><namespace>/drone/camera_back</namespace></ros>
            <camera_name>back</camera_name><frame_name>camera_back_link</frame_name>
          </plugin>
        </sensor>
        <sensor name="camera_left" type="camera">
          <pose>0 0.18 0 0 0 1.5708</pose>
          <camera><horizontal_fov>1.4</horizontal_fov><image><width>320</width><height>240</height></image><clip><near>0.1</near><far>100</far></clip></camera>
          <always_on>true</always_on><update_rate>10</update_rate><visualize>false</visualize>
          <plugin name="cam_left_plugin" filename="libgazebo_ros_camera.so">
            <ros><namespace>/drone/camera_left</namespace></ros>
            <camera_name>left</camera_name><frame_name>camera_left_link</frame_name>
          </plugin>
        </sensor>
        <sensor name="camera_right" type="camera">
          <pose>0 -0.18 0 0 0 -1.5708</pose>
          <camera><horizontal_fov>1.4</horizontal_fov><image><width>320</width><height>240</height></image><clip><near>0.1</near><far>100</far></clip></camera>
          <always_on>true</always_on><update_rate>10</update_rate><visualize>false</visualize>
          <plugin name="cam_right_plugin" filename="libgazebo_ros_camera.so">
            <ros><namespace>/drone/camera_right</namespace></ros>
            <camera_name>right</camera_name><frame_name>camera_right_link</frame_name>
          </plugin>
        </sensor>
      </link>

      <plugin name="planar_move" filename="libgazebo_ros_planar_move.so">
        <ros><namespace>/drone</namespace></ros>
        <command_topic>cmd_vel</command_topic>
        <odometry_topic>odom</odometry_topic>
        <odometry_frame>odom</odometry_frame>
        <robot_base_frame>link</robot_base_frame>
      </plugin>
    </model>
"""

FAIR_OBJECTS_BLOCK = """
    <!-- Fairground objects (basic shapes only — no internet download needed) -->
    <model name="stall_1"><static>true</static><pose>5 5 0.75 0 0 0</pose><link name="link">
      <visual name="base"><geometry><box><size>1.2 1.2 1.5</size></box></geometry><material><ambient>0.8 0.2 0.2 1</ambient></material></visual>
      <visual name="roof"><pose>0 0 0.85 0 0 0</pose><geometry><box><size>1.5 1.5 0.15</size></box></geometry><material><ambient>1 1 1 1</ambient></material></visual>
      <collision name="collision"><geometry><box><size>1.2 1.2 1.5</size></box></geometry></collision>
    </link></model>

    <model name="stall_2"><static>true</static><pose>-5 5 0.75 0 0 0.4</pose><link name="link">
      <visual name="base"><geometry><box><size>1.2 1.2 1.5</size></box></geometry><material><ambient>0.2 0.6 0.9 1</ambient></material></visual>
      <visual name="roof"><pose>0 0 0.85 0 0 0</pose><geometry><box><size>1.5 1.5 0.15</size></box></geometry><material><ambient>1 1 1 1</ambient></material></visual>
      <collision name="collision"><geometry><box><size>1.2 1.2 1.5</size></box></geometry></collision>
    </link></model>

    <model name="stall_3"><static>true</static><pose>5 -5 0.75 0 0 -0.6</pose><link name="link">
      <visual name="base"><geometry><box><size>1.2 1.2 1.5</size></box></geometry><material><ambient>0.9 0.8 0.1 1</ambient></material></visual>
      <visual name="roof"><pose>0 0 0.85 0 0 0</pose><geometry><box><size>1.5 1.5 0.15</size></box></geometry><material><ambient>1 1 1 1</ambient></material></visual>
      <collision name="collision"><geometry><box><size>1.2 1.2 1.5</size></box></geometry></collision>
    </link></model>

    <model name="tent_1"><static>true</static><pose>-5 -5 1.0 0 0 0</pose><link name="link">
      <visual name="tent"><geometry><box><size>2.5 2.5 2.0</size></box></geometry><material><ambient>0.3 0.7 0.3 1</ambient></material></visual>
      <collision name="collision"><geometry><box><size>2.5 2.5 2.0</size></box></geometry></collision>
    </link></model>

    <model name="tree_1"><static>true</static><pose>7 0 0 0 0 0</pose><link name="link">
      <visual name="trunk"><pose>0 0 0.75 0 0 0</pose><geometry><cylinder><radius>0.15</radius><length>1.5</length></cylinder></geometry><material><ambient>0.4 0.2 0.1 1</ambient></material></visual>
      <visual name="canopy"><pose>0 0 1.8 0 0 0</pose><geometry><sphere><radius>0.9</radius></sphere></geometry><material><ambient>0.1 0.5 0.1 1</ambient></material></visual>
      <collision name="collision"><pose>0 0 0.75 0 0 0</pose><geometry><cylinder><radius>0.15</radius><length>1.5</length></cylinder></geometry></collision>
    </link></model>

    <model name="tree_2"><static>true</static><pose>-7 0 0 0 0 0</pose><link name="link">
      <visual name="trunk"><pose>0 0 0.75 0 0 0</pose><geometry><cylinder><radius>0.15</radius><length>1.5</length></cylinder></geometry><material><ambient>0.4 0.2 0.1 1</ambient></material></visual>
      <visual name="canopy"><pose>0 0 1.8 0 0 0</pose><geometry><sphere><radius>0.9</radius></sphere></geometry><material><ambient>0.1 0.5 0.1 1</ambient></material></visual>
      <collision name="collision"><pose>0 0 0.75 0 0 0</pose><geometry><cylinder><radius>0.15</radius><length>1.5</length></cylinder></geometry></collision>
    </link></model>

    <model name="tree_3"><static>true</static><pose>0 7 0 0 0 0</pose><link name="link">
      <visual name="trunk"><pose>0 0 0.75 0 0 0</pose><geometry><cylinder><radius>0.15</radius><length>1.5</length></cylinder></geometry><material><ambient>0.4 0.2 0.1 1</ambient></material></visual>
      <visual name="canopy"><pose>0 0 1.8 0 0 0</pose><geometry><sphere><radius>0.9</radius></sphere></geometry><material><ambient>0.1 0.5 0.1 1</ambient></material></visual>
      <collision name="collision"><pose>0 0 0.75 0 0 0</pose><geometry><cylinder><radius>0.15</radius><length>1.5</length></cylinder></geometry></collision>
    </link></model>

    <model name="tree_4"><static>true</static><pose>0 -7 0 0 0 0</pose><link name="link">
      <visual name="trunk"><pose>0 0 0.75 0 0 0</pose><geometry><cylinder><radius>0.15</radius><length>1.5</length></cylinder></geometry><material><ambient>0.4 0.2 0.1 1</ambient></material></visual>
      <visual name="canopy"><pose>0 0 1.8 0 0 0</pose><geometry><sphere><radius>0.9</radius></sphere></geometry><material><ambient>0.1 0.5 0.1 1</ambient></material></visual>
      <collision name="collision"><pose>0 0 0.75 0 0 0</pose><geometry><cylinder><radius>0.15</radius><length>1.5</length></cylinder></geometry></collision>
    </link></model>

    <model name="bench_1"><static>true</static><pose>1.5 -1.5 0.25 0 0 0.3</pose><link name="link">
      <visual name="seat"><geometry><box><size>1.2 0.4 0.1</size></box></geometry><material><ambient>0.5 0.3 0.15 1</ambient></material></visual>
      <collision name="collision"><geometry><box><size>1.2 0.4 0.1</size></box></geometry></collision>
    </link></model>

    <model name="bench_2"><static>true</static><pose>-1.5 1.5 0.25 0 0 -0.3</pose><link name="link">
      <visual name="seat"><geometry><box><size>1.2 0.4 0.1</size></box></geometry><material><ambient>0.5 0.3 0.15 1</ambient></material></visual>
      <collision name="collision"><geometry><box><size>1.2 0.4 0.1</size></box></geometry></collision>
    </link></model>

    <model name="lamp_1"><static>true</static><pose>4 -4 0 0 0 0</pose><link name="link">
      <visual name="pole"><pose>0 0 1.5 0 0 0</pose><geometry><cylinder><radius>0.05</radius><length>3.0</length></cylinder></geometry><material><ambient>0.2 0.2 0.2 1</ambient></material></visual>
      <visual name="light"><pose>0 0 3.05 0 0 0</pose><geometry><sphere><radius>0.15</radius></sphere></geometry><material><ambient>1 1 0.7 1</ambient></material></visual>
      <collision name="collision"><pose>0 0 1.5 0 0 0</pose><geometry><cylinder><radius>0.05</radius><length>3.0</length></cylinder></geometry></collision>
    </link></model>

    <model name="lamp_2"><static>true</static><pose>-4 4 0 0 0 0</pose><link name="link">
      <visual name="pole"><pose>0 0 1.5 0 0 0</pose><geometry><cylinder><radius>0.05</radius><length>3.0</length></cylinder></geometry><material><ambient>0.2 0.2 0.2 1</ambient></material></visual>
      <visual name="light"><pose>0 0 3.05 0 0 0</pose><geometry><sphere><radius>0.15</radius></sphere></geometry><material><ambient>1 1 0.7 1</ambient></material></visual>
      <collision name="collision"><pose>0 0 1.5 0 0 0</pose><geometry><cylinder><radius>0.05</radius><length>3.0</length></cylinder></geometry></collision>
    </link></model>
"""

FOOTER = """
  </world>
</sdf>
"""


# ---------------------------------------------------------------------
# Balloons — static colorful sphere clusters on strings, scattered
# around the fairground at varying heights.
# ---------------------------------------------------------------------

BALLOON_COLORS = [
    "0.9 0.1 0.1 1",   # red
    "0.1 0.4 0.9 1",   # blue
    "0.9 0.8 0.1 1",   # yellow
    "0.2 0.8 0.3 1",   # green
    "0.9 0.3 0.7 1",   # pink
    "0.6 0.2 0.8 1",   # purple
]


def generate_balloon_cluster(index: int, x: float, y: float, ground_z: float,
                              cluster_height: float, rng: random.Random) -> str:
    """One cluster of 3 balloons on a string, anchored near (x, y)."""
    n_balloons = 3
    parts = []
    for b in range(n_balloons):
        color = rng.choice(BALLOON_COLORS)
        ox = rng.uniform(-0.15, 0.15)
        oy = rng.uniform(-0.15, 0.15)
        oz = cluster_height + rng.uniform(-0.1, 0.2)
        parts.append(
            f'      <visual name="balloon_{b}"><pose>{ox:.2f} {oy:.2f} {oz:.2f} 0 0 0</pose>'
            f'<geometry><sphere><radius>0.25</radius></sphere></geometry>'
            f'<material><ambient>{color}</ambient></material></visual>'
        )
    balloons_xml = "\n".join(parts)

    return f"""
    <model name="balloon_cluster_{index}">
      <static>true</static>
      <pose>{x:.2f} {y:.2f} {ground_z:.2f} 0 0 0</pose>
      <link name="link">
        <visual name="string"><pose>0 0 {cluster_height / 2:.2f} 0 0 0</pose>
          <geometry><cylinder><radius>0.01</radius><length>{cluster_height:.2f}</length></cylinder></geometry>
          <material><ambient>0.3 0.3 0.3 1</ambient></material></visual>
{balloons_xml}
      </link>
    </model>
"""


def generate_balloons(n_clusters: int, arena_half: float, rng: random.Random) -> str:
    blocks = []
    for i in range(n_clusters):
        x = rng.uniform(-arena_half - 1, arena_half + 1)
        y = rng.uniform(-arena_half - 1, arena_half + 1)
        height = rng.uniform(2.0, 3.0)
        blocks.append(generate_balloon_cluster(i + 1, x, y, 0.0, height, rng))
    return "".join(blocks)


# ---------------------------------------------------------------------
# Birds — small static silhouettes (body + two wing boxes) scattered
# in the sky. Static, not animated (Gazebo actors need a mesh file for
# animation, and no bird mesh is available offline). See README note
# at the bottom of this file for how to animate them later if wanted.
# ---------------------------------------------------------------------

def generate_bird(index: int, x: float, y: float, z: float, yaw: float) -> str:
    return f"""
    <model name="bird_{index}">
      <static>true</static>
      <pose>{x:.2f} {y:.2f} {z:.2f} 0 0 {yaw:.2f}</pose>
      <link name="link">
        <visual name="body">
          <geometry><sphere><radius>0.12</radius></sphere></geometry>
          <material><ambient>0.15 0.15 0.15 1</ambient></material>
        </visual>
        <visual name="wing_left">
          <pose>0 0.18 0 0.3 0 0</pose>
          <geometry><box><size>0.05 0.35 0.08</size></box></geometry>
          <material><ambient>0.15 0.15 0.15 1</ambient></material>
        </visual>
        <visual name="wing_right">
          <pose>0 -0.18 0 -0.3 0 0</pose>
          <geometry><box><size>0.05 0.35 0.08</size></box></geometry>
          <material><ambient>0.15 0.15 0.15 1</ambient></material>
        </visual>
      </link>
    </model>
"""


def generate_birds(n_birds: int, arena_half: float, rng: random.Random) -> str:
    blocks = []
    for i in range(n_birds):
        x = rng.uniform(-arena_half - 2, arena_half + 2)
        y = rng.uniform(-arena_half - 2, arena_half + 2)
        z = rng.uniform(6.0, 9.0)  # above the drone's 4.5m altitude
        yaw = rng.uniform(-math.pi, math.pi)
        blocks.append(generate_bird(i + 1, x, y, z, yaw))
    return "".join(blocks)


# ---------------------------------------------------------------------
# Giant (Ferris) wheel — built entirely from primitives: a hub, a rim
# approximated as segments arranged in a circle, spokes, two support
# legs, and evenly spaced gondola cabins. Static (not spinning).
# ---------------------------------------------------------------------

def generate_ferris_wheel(x: float, y: float, radius: float = 3.5,
                           n_cabins: int = 8) -> str:
    hub_z = radius + 1.5  # hub height above ground

    # rim: approximate a circle with short box segments
    n_segments = 24
    rim_parts = []
    for i in range(n_segments):
        angle = 2 * math.pi * i / n_segments
        seg_x = radius * math.cos(angle)
        seg_z = radius * math.sin(angle)
        seg_len = 2 * math.pi * radius / n_segments * 1.15  # slight overlap
        rim_parts.append(
            f'      <visual name="rim_{i}"><pose>{seg_x:.2f} 0 {seg_z:.2f} 0 {angle + math.pi/2:.3f} 0</pose>'
            f'<geometry><box><size>{seg_len:.2f} 0.12 0.12</size></box></geometry>'
            f'<material><ambient>0.7 0.1 0.1 1</ambient></material></visual>'
        )
    rim_xml = "\n".join(rim_parts)

    # spokes from hub to rim
    n_spokes = 8
    spoke_parts = []
    for i in range(n_spokes):
        angle = 2 * math.pi * i / n_spokes
        spoke_parts.append(
            f'      <visual name="spoke_{i}"><pose>0 0 0 0 {angle + math.pi/2:.3f} 0</pose>'
            f'<geometry><cylinder><radius>0.04</radius><length>{radius * 2:.2f}</length></cylinder></geometry>'
            f'<material><ambient>0.6 0.6 0.6 1</ambient></material></visual>'
        )
    spokes_xml = "\n".join(spoke_parts)

    # cabins evenly spaced around the rim
    cabin_parts = []
    for i in range(n_cabins):
        angle = 2 * math.pi * i / n_cabins
        cab_x = radius * math.cos(angle)
        cab_z = radius * math.sin(angle)
        color = BALLOON_COLORS[i % len(BALLOON_COLORS)]
        cabin_parts.append(
            f'      <visual name="cabin_{i}"><pose>{cab_x:.2f} 0 {cab_z:.2f} 0 0 0</pose>'
            f'<geometry><box><size>0.4 0.5 0.4</size></box></geometry>'
            f'<material><ambient>{color}</ambient></material></visual>'
        )
    cabins_xml = "\n".join(cabin_parts)

    return f"""
    <model name="giant_wheel">
      <static>true</static>
      <pose>{x:.2f} {y:.2f} {hub_z:.2f} 0 0 0</pose>
      <link name="link">
        <visual name="hub">
          <geometry><cylinder><radius>0.25</radius><length>0.3</length></cylinder></geometry>
          <material><ambient>0.4 0.4 0.4 1</ambient></material>
        </visual>
{spokes_xml}
{rim_xml}
{cabins_xml}
      </link>
    </model>

    <model name="giant_wheel_leg_1">
      <static>true</static>
      <pose>{x - 1.0:.2f} {y:.2f} {hub_z / 2:.2f} 0 0.35 0</pose>
      <link name="link">
        <visual name="leg"><geometry><cylinder><radius>0.15</radius><length>{hub_z * 1.1:.2f}</length></cylinder></geometry>
          <material><ambient>0.3 0.3 0.3 1</ambient></material></visual>
        <collision name="collision"><geometry><cylinder><radius>0.15</radius><length>{hub_z * 1.1:.2f}</length></cylinder></geometry></collision>
      </link>
    </model>

    <model name="giant_wheel_leg_2">
      <static>true</static>
      <pose>{x + 1.0:.2f} {y:.2f} {hub_z / 2:.2f} 0 -0.35 0</pose>
      <link name="link">
        <visual name="leg"><geometry><cylinder><radius>0.15</radius><length>{hub_z * 1.1:.2f}</length></cylinder></geometry>
          <material><ambient>0.3 0.3 0.3 1</ambient></material></visual>
        <collision name="collision"><geometry><cylinder><radius>0.15</radius><length>{hub_z * 1.1:.2f}</length></cylinder></geometry></collision>
      </link>
    </model>
"""


# ---------------------------------------------------------------------
# Pedestrian generation — this is the part that scales with N.
# ---------------------------------------------------------------------

def generate_pedestrian_actor(index: int, n_pedestrians: int, arena_half: float,
                               rng: random.Random) -> str:
    """Builds one <actor> block that walks around a circular 'lane' —
    each pedestrian gets its own radius (spread out from a small inner
    circle to the full arena), so pedestrians mostly stay in their own
    ring instead of all crossing through the same central area (which
    is what caused the visual pile-up with fully random waypoints)."""
    n_waypoints = 6  # points around the circle
    time_step = rng.uniform(3.0, 5.0)

    # spread radii across pedestrians: smallest lane ~30% of arena_half,
    # largest lane ~95% of arena_half
    if n_pedestrians > 1:
        frac = 0.3 + 0.65 * (index - 1) / (n_pedestrians - 1)
    else:
        frac = 0.6
    radius = arena_half * frac

    start_angle = rng.uniform(0, 2 * math.pi)
    direction = rng.choice([1, -1])  # clockwise or counter-clockwise

    waypoint_lines = []
    for k in range(n_waypoints + 1):  # +1 to close the loop back to start
        angle = start_angle + direction * (2 * math.pi * k / n_waypoints)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        # face along the direction of travel (tangent to the circle)
        yaw = angle + direction * (math.pi / 2)
        t = k * time_step
        waypoint_lines.append(
            f'          <waypoint><time>{t:.1f}</time>'
            f'<pose>{x:.2f} {y:.2f} 1.0 0 0 {yaw:.2f}</pose></waypoint>'
        )
    waypoints_xml = "\n".join(waypoint_lines)

    start_x = radius * math.cos(start_angle)
    start_y = radius * math.sin(start_angle)
    return f"""
    <actor name="pedestrian_{index}">
      <pose>{start_x:.2f} {start_y:.2f} 1.0 0 0 0</pose>
      <skin><filename>walk.dae</filename><scale>1.0</scale></skin>
      <animation name="walking"><filename>walk.dae</filename><scale>1.0</scale><interpolate_x>true</interpolate_x></animation>
      <script><loop>true</loop><auto_start>true</auto_start>
        <trajectory id="0" type="walking">
{waypoints_xml}
        </trajectory>
      </script>
    </actor>
"""


def generate_world(n_pedestrians: int, arena_half: float, seed: int,
                    n_balloon_clusters: int = 5, n_birds: int = 10,
                    add_ferris_wheel: bool = True) -> str:
    rng = random.Random(seed)
    pedestrian_blocks = "".join(
        generate_pedestrian_actor(i + 1, n_pedestrians, arena_half, rng)
        for i in range(n_pedestrians)
    )
    balloon_blocks = generate_balloons(n_balloon_clusters, arena_half, rng)
    bird_blocks = generate_birds(n_birds, arena_half, rng)
    ferris_wheel_block = (
        generate_ferris_wheel(x=-9.0, y=-9.0, radius=3.5, n_cabins=8)
        if add_ferris_wheel else ""
    )
    return (
        HEADER + DRONE_BLOCK + FAIR_OBJECTS_BLOCK + ferris_wheel_block
        + balloon_blocks + bird_blocks + pedestrian_blocks + FOOTER
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_pedestrians", type=int, default=6,
        help="How many pedestrians to generate (just increase this to grow the crowd)",
    )
    parser.add_argument(
        "--arena_half", type=float, default=3.5,
        help="Pedestrians roam within [-arena_half, +arena_half] on x and y",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible layouts")
    parser.add_argument("--output", type=str, default="crowd_world.world")
    parser.add_argument("--n_balloons", type=int, default=5, help="Number of balloon clusters")
    parser.add_argument("--n_birds", type=int, default=10, help="Number of static birds in the sky")
    parser.add_argument(
        "--no_ferris_wheel", action="store_true",
        help="Skip generating the giant wheel",
    )
    args = parser.parse_args()

    world_xml = generate_world(
        args.n_pedestrians, args.arena_half, args.seed,
        n_balloon_clusters=args.n_balloons,
        n_birds=args.n_birds,
        add_ferris_wheel=not args.no_ferris_wheel,
    )
    with open(args.output, "w") as f:
        f.write(world_xml)

    print(
        f"Generated {args.output} with {args.n_pedestrians} pedestrians, "
        f"{args.n_balloons} balloon clusters, {args.n_birds} birds, "
        f"ferris wheel={'yes' if not args.no_ferris_wheel else 'no'}."
    )


if __name__ == "__main__":
    main()
