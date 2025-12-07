# RoboRacer Competition Driver

## Architecture (FSAE Style)

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  LOCAL PLANNER  │───>│ GLOBAL PLANNER  │───>│  MPC CONTROLLER │
│  (Gap Follower) │    │ (Map + Path)    │    │  (Path Follow)  │
│  Always safety  │    │ Publishes map   │    │ After map ready │
└─────────────────┘    └─────────────────┘    └─────────────────┘
       ↓                      ↓                      ↓
  /cmd_local           /occupancy_grid         /mpc_active
  /emergency_stop      /global_path
                       /map_ready
```

---

## 🏁 COMPETITION Track (New Release!)

### Terminal 1 - Competition Simulator
```bash
xhost +local:root
docker run --name roboracer_sim_comp \
  --rm -it \
  --net=host \
  --gpus all \
  --privileged \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  autodriveecosystem/autodrive_roboracer_sim:2025-cdc-tf-compete
```
Then inside:
```bash
./AutoDRIVE\ Simulator.x86_64
```
*Click "Connection" once GUI opens*

### Terminal 2 - Competition Devkit + Bridge
```bash
xhost +local:root
docker run --name roboracer_devkit_comp \
  --rm -it \
  --net=host \
  --gpus all \
  --privileged \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v ~/roboracer_ws:/home/autodrive_devkit/roboracer_ws \
  autodriveecosystem/autodrive_roboracer_api:2025-cdc-tf-compete
```
Then inside:
```bash
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
ros2 launch autodrive_roboracer bringup_graphics.launch.py
```

### Terminal 3 - Build & Run (same devkit container)
```bash
docker exec -it roboracer_devkit_comp bash
```
Then inside:
```bash
# FIRST TIME ONLY: Install dependencies (see below)

source /opt/ros/humble/setup.bash
cd /home/autodrive_devkit/roboracer_ws
colcon build --packages-select wall_follower
source install/setup.bash

# Run your driver
ros2 run wall_follower slam_driver
```

---

## 🔧 First-Time Container Setup (Dependencies)

**Run this ONCE when starting a fresh container:**

```bash
# Install Python dependencies for track_mapper
pip3 install scikit-image scipy

# OR if pip3 doesn't work:
apt-get update && apt-get install -y python3-skimage python3-scipy
```

### What Each Dependency Does:
| Package | Used For |
|---------|----------|
| `scikit-image` | Skeletonization, path finding (`skimage.graph.route_through_array`) |
| `scipy` | Distance transform, morphological ops, smoothing |

### Quick One-Liner:
```bash
pip3 install scikit-image scipy && echo "✅ Dependencies installed!"
```

---

## 🏎️ PRACTICE Track (Qualification)

### Terminal 1 - Practice Simulator
```bash
xhost +local:root
docker run --name roboracer_sim \
  --rm -it \
  --net=host \
  --gpus all \
  --privileged \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  autodriveecosystem/autodrive_roboracer_sim:2025-cdc-tf-compete-practice
```
Then inside:
```bash
./AutoDRIVE\ Simulator.x86_64
```

### Terminal 2 - Practice Devkit + Bridge
```bash
xhost +local:root
docker run --name roboracer_devkit \
  --rm -it \
  --net=host \
  --gpus all \
  --privileged \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v ~/roboracer_ws:/home/autodrive_devkit/roboracer_ws \
  autodriveecosystem/autodrive_roboracer_api:2025-cdc-tf-practice
```
Then inside:
```bash
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
ros2 launch autodrive_roboracer bringup_graphics.launch.py
```

### Terminal 3 - Build & Run (same devkit container)
```bash
docker exec -it roboracer_devkit bash
```
Then inside:
```bash
# FIRST TIME ONLY: Install dependencies
pip3 install scikit-image scipy

source /opt/ros/humble/setup.bash
cd /home/autodrive_devkit/roboracer_ws
colcon build --packages-select wall_follower
source install/setup.bash

# Option A: Simple gap follower (qualification)
ros2 run wall_follower slam_driver

# Option B: Track mapper with pure pursuit
ros2 run wall_follower track_mapper

# Option C: Full racing stack with MPC + RViz
ros2 launch wall_follower racing.launch.py
```

---

## Quick Rebuild + Run (after containers already running)

```bash
docker exec -it roboracer_devkit bash -c "source /opt/ros/humble/setup.bash && cd /home/autodrive_devkit/roboracer_ws && colcon build --packages-select wall_follower && source install/setup.bash && ros2 run wall_follower slam_driver"
```

---

## Drivers Available

| Driver | Command | Description |
|--------|---------|-------------|
| `slam_driver` | `ros2 run wall_follower slam_driver` | Simple gap follower (qualification) |
| `racing.launch.py` | `ros2 launch wall_follower racing.launch.py` | Full stack: local+global+MPC+RViz |

---

## Dockerize for Submission

### Step 1: Start fresh container
```bash
docker run -it --name submission --entrypoint /bin/bash \
  autodriveecosystem/autodrive_roboracer_api:2025-cdc-tf-compete
```

### Step 2: Copy code into container (new terminal)
```bash
docker cp ~/roboracer_ws/src/wall_follower submission:/home/autodrive_devkit/src/
```

### Step 3: Inside container - build
```bash
cd /home/autodrive_devkit
source /opt/ros/humble/setup.bash
colcon build --packages-select wall_follower
```

### Step 4: Edit entrypoint script
```bash
nano /home/autodrive_devkit.sh
```

Paste this content:
```bash
#!/bin/bash

source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
cd /home/autodrive_devkit

# Launch bridge (headless)
ros2 launch autodrive_roboracer bringup_headless.launch.py &
sleep 2

# Launch racing algorithm
ros2 run wall_follower slam_driver &

wait
```

### Step 5: Commit and push (new terminal)
```bash
docker ps  # Get container ID

docker commit --change='CMD ["/home/autodrive_devkit.sh"]' \
  -m "Competition driver" -a "pranav" \
  submission pranman1/roboracer:competition

docker push pranman1/roboracer:competition
```

### Step 6: Test the container
```bash
# Start simulator first (Terminal 1)

# Then run your container:
docker run --rm -it --network=host pranman1/roboracer:competition

# Click "Connection" in simulator
```

---

## Competition Links

- Submit: `pranman1/roboracer:competition`
- Qualification: `pranman1/roboracer:qualification`

---

## Vehicle Parameters (Official)

| Parameter | Value |
|-----------|-------|
| Wheelbase | 0.3240 m |
| Max Steering | ±0.5236 rad (±30°) |
| Steering Rate | 3.2 rad/s |
| Top Speed | 22.88 m/s |
| Mass | 3.906 kg |
| LIDAR FOV | 270° (1080 points) |
