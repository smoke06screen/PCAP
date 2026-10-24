# FCV Code Walkthrough

The project has one perception ROS node. ROS 2 delivers an RGB camera frame to
`camera_test.py`; that node converts the message buffer into a NumPy array and
calls `process_traversability_pipeline()` in `perception.py`. The function
returns the detection result and seven image arrays. Processing is synchronous
and stateless: every frame is independent.

```text
Gazebo camera
  -> /camera/image_raw (sensor_msgs/Image, rgb8)
  -> camera_test.py
  -> perception.py
  -> result, debug images, and final masks
  -> log and optional OpenCV window
```

## File responsibilities

| File | Responsibility |
| --- | --- |
| `fcv/fcv/camera_test.py` | ROS node, image-message validation, display, and status logging |
| `fcv/fcv/perception.py` | Complete serial NumPy/OpenCV perception pipeline |
| `fcv/test/test_perception.py` | Synthetic and recorded-frame regression tests |
| `fcv/setup.py` | Python package, ROS entry point, and installed assets |
| `fcv/launch/bot.launch.py` | Gazebo, robot-state publisher, and robot spawning |

## ROS image handling

The callback accepts only `rgb8`. It checks positive dimensions, verifies that
`msg.step` can contain `width × 3` bytes, and verifies the buffer length. It
then removes optional row padding and makes a contiguous
`(height, width, 3)` `uint8` array.

The `show_window` parameter defaults to `true`. The display panels are
Original RGB, Blurred, Edges, Hough Lines, Boundary Mask, and Traversable Mask.
The node logs only when the validity state changes. The camera subscription
uses best-effort, keep-last QoS with depth one, so an older frame cannot build
up behind a newer one. The callback stores the newest display data; a separate
30 Hz timer composes the panels into one lightweight OpenCV window.

## Pipeline stages

All selected mask pixels have value 255; unselected pixels have value 0.
OpenCV's thread count is fixed to one, and the custom algorithm loops are
serial.

### 1. Grayscale

`rgb8_to_gray()` computes:

```text
0.299 × R + 0.587 × G + 0.114 × B
```

It adds 0.5 before conversion to `uint8`, preserving the original rounding.

### 2. Blur and Sobel display

`cv2.blur` applies a 3×3 box blur with replicated borders. `cv2.Sobel`
computes horizontal and vertical gradients. A pixel is an edge when
`abs(Gx) + abs(Gy) > 120`; outer image-border pixels are cleared.

### 3. Road candidate mask

For each pixel, the code computes brightness from grayscale, saturation as
`max(R,G,B) - min(R,G,B)`, and integer green excess as `G - (R+B)/2`. A pixel
is accepted when:

```text
35 < brightness < 215
saturation < 90
green_excess < 18
pixel is outside the visible-robot region
```

The visible-robot exclusion begins at 78% of image height. Its half-width grows
linearly from 12% to 21% of image width toward the bottom.

### 4. Sparse boundary evidence

Rows from 25% through just below 78% of image height form the ROI. On each row,
the algorithm starts at the center and finds the left and right ends of the
connected road-mask run. Runs narrower than one twelfth of the image width are
ignored. Endpoints that do not touch an image border become Hough evidence and
are also added to the Edges diagnostic image.

Hough voting uses this sparse evidence, not every Sobel edge.

### 5. Hough candidates

Each evidence point votes in 180 angle bins using
`rho = x × cos(theta) + y × sin(theta)`. A candidate must:

- be a local accumulator peak within ±2 angle and ±3 rho bins;
- have votes from at least 12% of ROI rows;
- satisfy `abs(cos(theta)) >= 0.35`;
- start between 12% and 88% of image width at the ROI top;
- be visible for at least one fifth of ROI rows; and
- have evidence within ±2 pixels on at least 45% of visible rows.

Up to 24 candidates per side are retained. Candidate score is:

```text
0.55 × orientation + 0.35 × support + 0.10 × normalized votes
```

Orientation favors a modest outward slant of about 0.35 horizontal pixels per
row and penalizes a boundary that leans the wrong way for its side.

### 6. Pair selection

Every retained left/right combination is checked. Geometry rejects a pair if
too few rows show both lines, if widths are outside the fixed limits, or if the
pair shrinks too strongly. Top width must be 8%–90% of image width. Width at
the final jointly visible row must be 18%–98% and at least 75% of top width.

The pair score is:

```text
0.40 × left score + 0.40 × right score + 0.20 × geometry
```

The best pair is accepted at a score of at least 0.65. Accepted lines are drawn
in `boundary_mask`. `traversable_mask` is filled between them only where the
road mask is also accepted, then both final masks have the visible-robot region
cleared.

## Result states

| Value | Meaning | Traversable mask |
| ---: | --- | --- |
| 0 | No reliable boundary | Empty |
| 1 | Left boundary only | Empty |
| 2 | Right boundary only | Empty |
| 3 | Accepted boundary pair | Filled where road pixels pass |
| 4 | Candidates exist but every pair fails geometry | Empty |
| 5 | A valid pair scores below 0.65 | Empty |

`confidence`, `left_confidence`, and `right_confidence` preserve single-precision
result values. They are hand-designed evidence weights, not probabilities.

## Build, run, and test

From the workspace root:

```bash
cd ~/PCAP
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select fcv
source install/setup.bash
```

Run Gazebo and perception in separate sourced terminals:

```bash
ros2 launch fcv bot.launch.py
```

```bash
ros2 run fcv camera_test
# Headless:
ros2 run fcv camera_test --ros-args -p show_window:=false
```

Run focused tests from `~/PCAP/src`:

```bash
python3 -m pytest -q fcv/test/test_perception.py
```

The perception tests include exact output hashes and result values for all
three recorded frames. Separate message-conversion tests cover padded rows,
invalid dimensions, and non-RGB8 encoding. Gazebo does not need to be running.

## Known limits

- Straight lines cannot model curved boundaries exactly.
- Similar road and background colors can merge in the road mask.
- The fixed ROI does not adapt to camera pitch or a measured horizon.
- The package has no temporal tracking or autonomous steering.
