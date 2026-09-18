# Consistent ROS2 Humble + Gazebo 11 environment for the crowd_drone_sim project.
# Ensures every teammate runs the exact same ROS2/Gazebo versions, avoiding the
# kind of version-mismatch issues (numpy, cv2, etc.) encountered during development.

FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-c"]

# --- System packages ---
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-plugins \
    && rm -rf /var/lib/apt/lists/*

# --- Python packages used by the bridge/policy-runner scripts ---
RUN pip3 install --upgrade numpy \
    && pip3 install stable-baselines3 gymnasium

# --- Project files ---
WORKDIR /workspace
COPY drone_ws /workspace/drone_ws

# --- Build the ROS2 workspace ---
RUN source /opt/ros/humble/setup.bash \
    && cd /workspace/drone_ws \
    && colcon build

# --- Auto-source ROS2 + workspace on container start ---
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "source /workspace/drone_ws/install/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
