# Docker GUI Notes (Gazebo passthrough)

Gazebo needs a graphical display, which Docker containers don't have by default.
Setup differs slightly depending on each teammate's OS:

## Windows 11 with WSL2 + Docker Desktop

WSL2 on Windows 11 includes WSLg (built-in GUI support), and Docker Desktop's
WSL2 backend passes this through automatically in most setups. Usually works
out of the box with the `docker-compose.yml` in this folder — just run:

```bash
docker compose up --build
docker exec -it crowd_drone_sim bash
```

If the Gazebo window doesn't appear, try setting `DISPLAY` explicitly before
`docker compose up`:
```bash
export DISPLAY=:0
```

## Native Linux

Allow the container to connect to your X server first:
```bash
xhost +local:docker
docker compose up --build
docker exec -it crowd_drone_sim bash
```

## macOS

Requires XQuartz (X11 server for macOS) installed and configured to allow
network connections. More setup-intensive — if a Mac teammate hits friction
here, it's usually easier for them to just run the pure-Python parts
(`traj_predictor/`, `drone_rl/`) locally and rely on a Windows/Linux teammate
for Gazebo-specific work.

## Running things once inside the container

```bash
source /opt/ros/humble/setup.bash
source /workspace/drone_ws/install/setup.bash
cd /workspace/drone_ws/src/crowd_drone_sim/worlds
python3 generate_world.py --n_pedestrians 6
gazebo crowd_world.world
```

## If you don't need the GUI (e.g. running a headless training/eval script)

Skip the X11 mounting entirely and just use `gzserver` (physics only, no
rendering) instead of `gazebo` — much simpler, no display setup needed at all.
