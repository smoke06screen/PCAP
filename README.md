# FCV — Camera-Based Traversability Perception

FCV is a ROS 2 Humble and Gazebo Classic project that estimates straight road
boundaries and a traversable image region for a simulated mobile robot. The
perception pipeline is implemented in serial Python using NumPy and OpenCV.

See [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) for the algorithm and code
layout.

## Current pipeline

One ROS node, `camera_test`, subscribes to `/camera/image_raw`, processes each
`rgb8` frame, reports the boundary state, and optionally shows six diagnostic
panels in a lightweight OpenCV window. The subscription keeps only the newest
sensor frame, and display refresh is handled by a separate timer. The stages
are:

1. RGB-to-grayscale conversion.
2. 3×3 box blur.
3. Thresholded Sobel edges.
4. Color-based road candidate mask.
5. Center-connected boundary evidence.
6. Custom Hough voting and line scoring.
7. Left/right pair selection and traversable-mask filling.

The fixed thresholds, ROI, Hough parameters, scoring, geometry checks, and
validity values are documented in the walkthrough. The detector is a
straight-boundary prototype and its confidence is an evidence score, not a
calibrated probability.

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic with `gazebo_ros` and `gazebo_plugins`
- Python 3, NumPy, and OpenCV

The installed `camera_test` command disables per-user Python packages before
loading the node. This prevents packages installed with `pip --user` (such as
NumPy 2) from overriding the ROS Humble-compatible Ubuntu packages.

Install package dependencies from the workspace root:

```bash
cd ~/PCAP
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
```

No C compiler or OpenMP runtime is required by this branch.

## Build

Run from the workspace root, not from `src`:

```bash
cd ~/PCAP
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select fcv
source install/setup.bash
```

Re-run the final `source install/setup.bash` in each new terminal after sourcing
ROS 2.

## Run

Start Gazebo and spawn the robot:

```bash
cd ~/PCAP
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch fcv bot.launch.py
```

In a second sourced terminal, start the perception node:

```bash
cd ~/PCAP
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run fcv camera_test
```

For a machine without a display, or to disable the diagnostic window:

```bash
ros2 run fcv camera_test --ros-args -p show_window:=false
```

The node expects `/camera/image_raw` with `rgb8` encoding. Useful checks are:

```bash
ros2 topic list | grep camera
ros2 topic info /camera/image_raw
ros2 topic hz /camera/image_raw
```

## Test

From the repository root:

```bash
cd ~/PCAP/src
python3 -m pytest -q fcv/test
```

The perception tests cover blank, single-boundary, paired, narrow,
horizontal-stripe, slanted, and tiny images, plus recorded Gazebo frames. They
also lock every output byte and result field for all three recorded frames.
Camera-message tests cover padding, encoding, and invalid dimensions. Gazebo
does not need to be running.

## Offline benchmark

To time the serial Python pipeline on the recorded frames:

```bash
cd ~/PCAP/src
python3 bench/benchmark_python.py --repetitions 20 --warmup 3
```

The timer excludes ROS, Gazebo, plotting, and frame loading. The old files in
`bench/results/native_*` are retained only as historical measurements from the
former C/OpenMP implementation; they do not describe this branch's runtime.

## Main files

- `fcv/fcv/camera_test.py`: ROS subscriber, message validation, logging, and
  display.
- `fcv/fcv/perception.py`: complete serial perception algorithm.
- `fcv/test/test_perception.py`: synthetic and recorded-frame regression tests.
- `fcv/launch/bot.launch.py`: Gazebo and robot launch.
- `fcv/urdf/fcv_model.urdf`: robot and sensor description.
- `bench/benchmark_python.py`: offline serial timing utility.

## Current limits

- Each boundary is modeled as a straight line.
- The road classifier is a fixed color heuristic tuned for the current Gazebo
  scene.
- The ROI is fixed at 25%–78% of image height.
- There is no temporal tracking, steering, or published perception output
  topic; the result is currently logged and displayed.
- Perception quality has not been measured against a labelled dataset.
