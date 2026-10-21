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

Implemented components:

-   ROS2 Python package named `pcap`.
-   Gazebo robot description and launch entry point.
-   Camera subscriber in `camera_test.py`.
-   Direct handling of `sensor_msgs/Image.data` for `rgb8` images using
    NumPy.
-   Native C shared library loaded through Python `ctypes`.
-   RGB-to-grayscale, 3×3 box blur, thresholded Sobel edges, a color
    candidate mask, and Hough voting in C using OpenMP where applicable.
-   Hough candidates scored by vertical orientation, edge continuity,
    and votes; a compatible left/right pair is selected before filling
    the traversable mask.
-   Explicit detector states and evidence scores available from the C
    library and shown by the ROS node.
-   Shared library compiled with GCC and linked against the OpenMP
    runtime.
-   Six-panel Matplotlib diagnostic display.

The detector remains a simple straight-boundary prototype. It has not
been validated against labelled scenes. Its score is not a calibrated
probability. Serial C variants, an offline benchmark runner, controlled
scenario datasets, and quality measurements are still to be added.

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
sudo apt install build-essential
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

The native C/OpenMP library is compiled automatically during the
`colcon` build.

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

Run the registered camera-processing executable:

``` bash
ros2 run pcap camera_test
```

On a headless system, disable the Matplotlib window:

``` bash
ros2 run pcap camera_test --ros-args -p show_window:=false
```

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

The node accepts `rgb8`, validates the row stride and buffer size, and
copies the active pixels into a contiguous NumPy array before calling C.
Conceptually:

``` python
rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
image = np.ascontiguousarray(
    rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
)
```

### C/OpenMP interface

The Python wrapper passes input and output buffers through `ctypes`. The
C function should receive explicit dimensions and should not retain
pointers after the call returns.

The native pipeline currently uses OpenMP for pixel loops and Hough
voting. Small images or lightweight kernels may not benefit from
parallelization because thread-management overhead can outweigh the
work. The current code has no serial C counterpart, so no serial versus
OpenMP performance claim can yet be made.

### Boundary selection and failure states

Hough votes come from thresholded Sobel edges inside the color candidate
mask, between 30% and 78% of image height. Candidate lines are checked
for edge support within two pixels along this region. Lines with support
below 0.45 are rejected. The candidate score is
`0.55 × verticality + 0.35 × edge support + 0.10 × normalized votes`,
where `verticality = |cos(theta)|` for the Hough line normal. This gives
near-vertical lines the most weight; lines with verticality below 0.35
are rejected, so near-horizontal markings cannot become boundaries.
These constants are currently fixed in C and should be tuned only with
recorded test data.

The detector tests left/right pairs for ordering, plausible width at
the top and bottom of the region, and convergence. A valid pair fills
the pixels between the lines where the color candidate mask accepts
them, excluding the visible robot region. A single reliable boundary
is displayed but produces no traversable mask. Rejected or missing
pairs also produce an empty traversable mask. The status codes are:
`0` no boundary, `1` left only, `2` right only, `3` valid pair,
`4` pair rejected by geometry, and `5` low-confidence pair.
The score is an interpretable evidence score in `[0, 1]`, not a
probability of correctness.

The existing `process_traversability_pipeline` C entry point remains
available. `process_traversability_pipeline_ex` takes the same arguments
plus a final pointer to a result structure containing `int validity`
followed by `float confidence`, `float left_confidence`, and
`float right_confidence`. Both entry points return `0` for successful
processing or a negative error code. The ROS node uses the extended
entry point.

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

### `libperception*.so` cannot be loaded

-   Install GCC/OpenMP build support and rebuild the package.
-   Run `ldd` on the installed library and resolve any `not found`
    dependencies.
-   Source the rebuilt workspace before starting the node.

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
