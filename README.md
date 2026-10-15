# PCAP --- Parallelized Camera-Based Traversability Perception

A ROS2 and Gazebo Classic project exploring lightweight road-boundary
and traversable-region detection for a simulated mobile robot, with
image-processing kernels implemented in C and parallelized using OpenMP.

## Overview

PCAP investigates whether a classical computer-vision pipeline can
estimate road boundaries and a traversable region from a monocular
camera while keeping computationally intensive image processing in
native C.

The intended evaluation compares **serial C** and **OpenMP-parallel C**
implementations on identical frames. It will measure computational
performance and perception quality under controlled visual conditions,
including shadows, weak road boundaries, surface patches, and curved
roads.

## Goals

-   Ingest simulated camera images through ROS2.
-   Keep performance-critical image-processing kernels in C.
-   Use OpenMP for parallelizable loops.
-   Keep Python as a thin ROS2, buffer-handling, library-call, and
    visualization layer.
-   Develop road-boundary and traversable-region estimation.
-   Compare serial and parallel implementations using identical inputs.
-   Evaluate robustness on repeatable visual test cases.

## Current implementation

Established components:

-   ROS2 Python package named `pcap`.
-   Gazebo robot description and launch entry point.
-   Camera subscriber in `camera_test.py`.
-   Direct handling of `sensor_msgs/Image.data` for `rgb8` images using
    NumPy.
-   Native C shared library loaded through Python `ctypes`.
-   RGB-to-grayscale conversion in C using an OpenMP parallel loop.
-   Shared library compiled with GCC and linked against the OpenMP
    runtime.
-   OpenCV display of the grayscale result.

## Architecture

``` text
Gazebo Classic camera sensor
          |
          | /camera/image_raw (sensor_msgs/msg/Image)
          v
ROS2 Python node
          |
          | Read rgb8 message data directly
          | NumPy buffer/view
          v
ctypes call into native shared library
          |
          v
C image-processing kernel(s)
          |
          | OpenMP for parallelizable loops
          v
Output NumPy buffer
          |
          v
Python/OpenCV visualization and evaluation
```

The Python layer is intended to remain thin. Native image-processing
kernels can then be benchmarked independently of ROS message handling
and visualization.

## Requirements

The development environment used for this project includes:

-   Ubuntu 22.04
-   ROS2 Humble
-   Gazebo Classic
-   Python 3
-   NumPy
-   OpenCV Python bindings (`cv2`)
-   GCC with OpenMP support
-   `ctypes` (Python standard library)

The native library is compiled with `-fopenmp`. On Linux, it commonly
links against the OpenMP runtime `libgomp.so.1`.

## Build

From the workspace root:

``` bash
cd ~/PCAP
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Compile the native library from the package source directory:

``` bash
cd ~/PCAP/src/pcap

gcc -O3 -fopenmp -fPIC -shared \
    pcap/native/perception.c \
    -o pcap/native/libperception.so
```

Check the library and its OpenMP dependency:

``` bash
file pcap/native/libperception.so
ldd pcap/native/libperception.so | grep -E 'gomp|not found'
```

If the package installs the library into the workspace's install tree,
rebuild the package after recompiling:

``` bash
cd ~/PCAP
colcon build --symlink-install --packages-select pcap
source install/setup.bash
```

Make sure the path passed to `ctypes.CDLL` points to the current
library. Avoid relying on the current working directory, since ROS2
launch files may start a node from a different directory.

## Running

### 1. Source the environment

``` bash
source /opt/ros/humble/setup.bash
cd ~/PCAP
source install/setup.bash
```

### 2. Launch the robot and simulation

The package contains `launch/bot.launch.py`:

``` bash
ros2 launch pcap bot.launch.py
```

The exact launch behavior depends on the current launch file and local
Gazebo setup.

### 3. Inspect camera topics

In another sourced terminal:

``` bash
ros2 topic list | grep camera
ros2 topic info /camera/image_raw
ros2 topic hz /camera/image_raw
```

The expected raw image topic is `/camera/image_raw`; the current sensor
configuration expects `/camera/camera_info` for camera information.
Confirm actual topic names with `ros2 topic list`.

### 4. Run the camera-processing node

The package contains `camera_test.py`. If it is registered as a console
script in `setup.py`, run:

``` bash
ros2 run pcap camera_test
```

If it is not registered as a console script, use the package's current
development workflow or add the appropriate `console_scripts` entry to
`setup.py`.

## Camera configuration

The current camera configuration uses a 640 × 480 RGB image, a
configured update rate of 10 Hz, and horizontal FOV specified in
radians.

Example values:

``` xml
<camera>
  <horizontal_fov>1.5708</horizontal_fov>
  <image>
    <format>R8G8B8</format>
    <width>640</width>
    <height>480</height>
  </image>
  <clip>
    <near>0.05</near>
    <far>15.0</far>
  </clip>
</camera>
```

`1.5708` radians is approximately 90 degrees. The standard Gazebo
Classic camera configuration specifies horizontal FOV; vertical FOV
follows from the image aspect ratio. These are example values, not
mandatory settings. A wider FOV captures more of the scene but makes
each road feature occupy fewer pixels at a fixed resolution.

## Implementation notes

### ROS image buffer

The prototype assumes `rgb8`. Conceptually:

``` python
image = np.frombuffer(msg.data, dtype=np.uint8)
image = image.reshape((msg.height, msg.width, 3))
```

Validate `msg.encoding`, dimensions, and buffer length before using this
pattern. Do not assume every ROS image encoding has three channels.

### C/OpenMP interface

The Python wrapper passes input and output buffers through `ctypes`. The
C function should receive explicit dimensions and should not retain
pointers after the call returns.

The current native operation is RGB-to-grayscale conversion. Its OpenMP
loop uses static scheduling and is compiled with optimization and OpenMP
enabled. Small images or lightweight kernels may not benefit from
parallelization because thread-management overhead can outweigh the
work.

### Thread count

For controlled experiments, set the OpenMP thread count explicitly:

``` bash
export OMP_NUM_THREADS=4
```

Record the thread count and machine details for each benchmark. Compare
against a serial C implementation using the same input, output
requirements, and timing method.

### Suggested metrics

**Performance** - Mean and median processing time per frame. - Per-stage
and end-to-end runtime. - Throughput in frames per second. - Speedup:
`serial runtime / OpenMP runtime`. - Parallel efficiency, when useful:
`speedup / thread count`.

**Perception** - Boundary-position error against labelled ground
truth. - Traversable-mask intersection over union (IoU). - Detection
success rate and false-detection rate. - Recovery behavior after
shadows, weak boundaries, or occlusion.

Document the measurement method, sample count, hardware, compiler flags,
thread count, and whether timings include ROS message handling or
visualization. Keep native-kernel benchmarks separate from end-to-end
pipeline timings.


## Troubleshooting

### `libperception.so` cannot be loaded

-   Recompile the library using the command above.
-   Check the path used by `ctypes.CDLL`.
-   Run `ldd` on the library and resolve any `not found` dependencies.
-   Rebuild the ROS2 package if the install tree contains an older
    copy.

### `cv_bridge` / NumPy ABI error

The current development environment has encountered incompatibility
between the installed ROS2 Humble `cv_bridge` and NumPy 2.x. The
prototype avoids `cv_bridge` by reading the `rgb8` buffer directly. If
the image conversion path changes, test the full ROS node and native
call.

### Camera topic is missing

-   Confirm Gazebo is running and the robot spawned.
-   Inspect `ros2 topic list`.
-   Check the sensor and plugin configuration in the URDF.
-   Verify the topic name and message type with `ros2 topic info`.

### Actual camera rate is lower than configured

The sensor's `<update_rate>` is a requested update rate, not a guarantee
that the whole simulation and subscriber pipeline will sustain it.
Measure and report the achieved rate.

### OpenMP is slower than serial

This can happen for small images or lightweight kernels. Measure kernel
time separately, control thread count, exclude unrelated display time
from kernel benchmarks, and report the result honestly.

## Acknowledgements

Built with ROS2, Gazebo Classic, OpenCV, NumPy, GCC, and OpenMP.
