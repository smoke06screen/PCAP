# PCAP code walkthrough

This guide explains the repository as it currently works. Start here if you have not read the code before. PCAP uses a simulated robot camera to estimate the two sides of a road and mark pixels that appear traversable. ROS 2 and Python handle incoming images and the display; native C does the image processing; OpenMP distributes several C loops across CPU threads.

The current detector is a **single-frame, straight-line prototype**. It does not drive the robot, track boundaries over time, or estimate a probability that a detection is correct.

## Follow one camera frame

```text
Gazebo world and robot camera (URDF)
    -> /camera/image_raw: sensor_msgs/Image, expected rgb8
    -> camera_test.py: validate and unpack the image
    -> NumPy arrays passed by pointer through ctypes
    -> perception.c: grayscale, blur, Sobel, color mask, boundary evidence,
       Hough candidates, pair selection, traversable mask
    -> C result structure and output arrays
    -> camera_test.py: log a status change and update six display panels
```

The ROS callback processes each received frame synchronously. The C function finishes before Python reads its outputs. Nothing in this pipeline uses a previous frame.

## Where things live

| Path | Purpose |
| --- | --- |
| `pcap/launch/bot.launch.py` | Starts Gazebo with this world, publishes the robot description, and spawns the robot. It does **not** start the perception node. |
| `pcap/worlds/pcap_test.world` | Includes the sun and the `igvc_ground` model. |
| `pcap/models/igvc_ground/` | Ground model, material script, and texture assets used by Gazebo. |
| `pcap/urdf/pcap_model.urdf` | Robot links, wheels, camera, Gazebo camera plugin, and differential-drive plugin. The lidar sensor block is commented out. |
| `pcap/pcap/camera_test.py` | ROS image subscriber, `ctypes` binding, buffers, logging, and Matplotlib display. |
| `pcap/pcap/native/perception.c` | Image processing, boundary selection, masks, confidence scores, and OpenMP directives. |
| `pcap/setup.py` and `pcap/package.xml` | Python package entry point, native extension build, installed assets, and ROS dependencies. |
| `pcap/test/test_perception.py` | Synthetic and saved-frame tests for the C library. |
| `pcap/test/frames/` | Three captured Gazebo RGB frames and accompanying diagnostic files. |
| `bench/benchmark_native.py` and `bench/RESULTS.md` | Offline serial C versus OpenMP timing runner and measured results. |

All paths above are relative to this repository root. `README.md` gives a shorter project overview and troubleshooting notes.

## Simulation and camera input

The launch file adds the packaged Gazebo models to `GAZEBO_MODEL_PATH`, starts `gazebo_ros` with `pcap_test.world`, starts `robot_state_publisher`, and spawns the URDF as `my_bot`. The robot's camera configuration in the URDF requests a 640 × 480 `R8G8B8` image at 10 Hz. Its horizontal field of view is `1.5708` radians, about 90 degrees, and its near/far clip distances are 0.05/15 m. The requested sensor rate is not a measurement of the achieved ROS topic rate or the processing rate.

The Gazebo camera plugin publishes the image expected by the Python node at `/camera/image_raw`. The node subscribes with a queue depth of 10. The differential-drive plugin also exists in the URDF, but the perception node does not command it.

The node accepts only ROS images whose encoding is exactly `rgb8`. It checks positive width and height, that the row stride (`msg.step`) can hold `width × 3` bytes, and that `msg.data` contains at least `msg.step × height` bytes. NumPy first views the message buffer as rows, drops any padding at the end of each row, then makes a contiguous `(height, width, 3)` byte array when needed. That layout is what the C code expects: RGB bytes in row-major order, with no padding.

For every accepted frame, Python allocates seven `(height, width)` `uint8` output arrays: `gray`, `denoised`, `edges`, `road_mask`, `hough_lines`, `boundary_mask`, and `traversable_mask`. It passes their pointers, the input pointer, image dimensions, and a result-structure pointer to C. The buffers remain alive until the call returns.

## Native C pipeline

The public entry point used by the node is `process_traversability_pipeline_ex(...)`. It returns `0` on success, `-1` for invalid arguments, and `-2` if an internal allocation fails. The C result structure and Python `ctypes.Structure` have the same field order: `int validity`, then three `float` values named `confidence`, `left_confidence`, and `right_confidence`.

The stages below run in this order. Unless stated otherwise, a mask uses `255` for selected pixels and `0` for everything else.

### 1. RGB to grayscale

For each pixel, C computes approximately `0.299 × R + 0.587 × G + 0.114 × B` and rounds to the nearest byte. The result is the `gray` array. This step always runs, even if the caller requests only grayscale conversion.

### 2. Smoothing

If a `denoised` buffer is supplied, C averages a 3 × 3 neighborhood of grayscale pixels. Neighbor coordinates at image borders are clamped to the nearest valid pixel. This is a **box blur**, not a Gaussian blur. The ROS node always supplies this buffer.

### 3. Sobel edge display

C applies 3 × 3 Sobel horizontal and vertical gradients to the blurred image (or directly to grayscale if no blur buffer was supplied). It sets an edge pixel to `255` when `abs(Gx) + abs(Gy) > 120`; image-border pixels are set to zero. The resulting `edges` array is useful for visual diagnosis.

**Important:** the Hough transform does not vote from all Sobel edge pixels. Later, the color-mask stage adds boundary points to the displayed `edges` array, but Hough voting uses a separate, sparse `boundary_evidence` array. A visible white line in the Edges panel therefore does not guarantee a Hough candidate.

### 4. Color-based road candidate mask

For each RGB pixel, C computes brightness from grayscale, saturation as `max(R,G,B) - min(R,G,B)`, and `green_excess = G - (R+B)/2` using integer arithmetic. A pixel enters `road_mask` only when:

```text
35 < brightness < 215
saturation < 90
green_excess < 18
pixel is outside the estimated visible-robot region
```

These constants are fixed in C, not ROS parameters. The mask is a color heuristic for the current scene, not a learned road classifier. Pale gravel can pass these tests alongside asphalt.

### 5. Boundary evidence from the center-connected mask

Only rows from 25% of image height through just below 78% are searched. For each row, C checks the mask at the image center. If that center pixel is road-like, it walks left and right until the connected run ends. Runs narrower than one twelfth of the image width are ignored. When a run ends before the image border, its left and/or right endpoint is marked in `boundary_evidence`. The same points are added to the displayed `edges` array.

This produces at most two voting points per accepted row. It reduces votes from asphalt texture and from the world boundary above the search region. It also creates a known failure case: if road and gravel have similar accepted colors, the connected run can extend to the image edge, leaving no right-hand boundary evidence even when a Sobel edge is visible.

The 25%–78% range is a fixed image-space region of interest (ROI), not a measured horizon. At 640 × 480 it covers rows 120 through 373. The bottom 22% is excluded from line voting because the robot body can appear there.

### 6. Hough voting and candidate filtering

For each nonzero boundary-evidence pixel inside the ROI, C votes over 180 line-angle bins using the normal-form equation `rho = x × cos(theta) + y × sin(theta)`. The shared accumulator has one cell for each angle and distance bin. A vote increment is protected by `#pragma omp atomic` because different threads may reach the same cell.

After voting, C searches for local accumulator peaks. A candidate must have at least 12% as many votes as the ROI has rows. Near-horizontal lines are excluded with `abs(cos(theta)) < 0.35`. The candidate's projected position at the top of the ROI must lie between 12% and 88% of image width. It must remain visible for at least one fifth of ROI rows, and evidence must support at least 45% of its visible rows within ±2 pixels. Up to 24 candidates per side are retained in score order.

The `hough_lines` debug image draws the highest-scoring candidate on each side. Those debug candidates are **not necessarily the pair finally selected**. The final selected lines appear in `boundary_mask`.

### 7. Candidate scores and pair selection

A candidate's score is:

```text
0.55 × orientation_score + 0.35 × support_fraction + 0.10 × vote_score
```

`vote_score` is votes divided by ROI height, capped at 1. The orientation score favors a modest outward slant toward the image bottom, peaking around 0.35 horizontal pixels per vertical pixel. A vertical line still receives a positive score; a line leaning the wrong way for its side is penalized. This is a preference, not a strict requirement that a boundary reach the bottom of the image.

C then tries left/right combinations. Pair geometry rejects combinations that have too little simultaneous visibility, implausible widths, reversed or strongly shrinking width, or poor ordering. In particular, top width must be 8%–90% of image width; width at the last mutually visible row must be 18%–98% and at least 75% of top width. A surviving pair receives a geometry score based on centering and width convergence. Its combined score is:

```text
0.40 × left_score + 0.40 × right_score + 0.20 × pair_geometry
```

The best pair is accepted only if that combined score is at least `0.65`. These scores are hand-designed evidence weights in `[0, 1]`; they have **not** been calibrated as probabilities of a correct detection.

### 8. Final masks and result state

For an accepted pair, C draws both lines into `boundary_mask`. Between those lines and within the ROI, it fills `traversable_mask` only where `road_mask` is also nonzero. Thus the output can have holes even inside the line pair. It then clears any pixels in the estimated visible-robot region from the final boundary and traversable masks.

If only one side has a candidate, C draws that side in `boundary_mask` but leaves `traversable_mask` empty. A rejected or low-confidence pair also leaves the traversable mask empty. No result silently marks the whole frame traversable.

| `validity` | Meaning | Traversable mask |
| ---: | --- | --- |
| 0 | No reliable boundary | Empty |
| 1 | Left boundary only | Empty |
| 2 | Right boundary only | Empty |
| 3 | Accepted left/right pair | Filled where the road mask accepts pixels |
| 4 | Left/right candidates exist, but geometry rejects every pair | Empty |
| 5 | A geometrically valid pair exists, but its score is below 0.65 | Empty |

## C entry points and display

`process_traversability_pipeline_ex` is the full interface described above. The older `process_traversability_pipeline` calls it but discards the result structure; it remains for callers using the earlier C signature. `process_rgb8_to_gray` calls the legacy interface with later output pointers set to `NULL`, so it runs grayscale conversion only. The ROS node binds the full and grayscale functions, but currently calls only the full `_ex` function for incoming camera frames.

The optional `show_window` ROS parameter defaults to `true`. When enabled, Matplotlib shows **Original RGB**, **Blurred**, **Edges**, **Hough Lines**, **Boundary Mask**, and **Traversable Mask**. There is no separate grayscale or road-mask panel. The title includes the validity label and overall score. The node logs status and score when the validity state changes; it does not log every frame. With `show_window:=false`, the C processing still runs but no figure is created. The node does not currently publish the detected boundaries or mask as ROS topics.

## Where OpenMP acts

The extension is built with `-O3` and `-fopenmp`; linking also uses `-fopenmp` and `-lm`. All OpenMP directives are in `perception.c`:

| Work | Directive |
| --- | --- |
| Clear the robot region, RGB-to-grayscale, box blur, Sobel edges, road-color mask, and traversable-mask fill | `#pragma omp parallel for schedule(static)` on the outer row loop |
| Hough voting | `#pragma omp parallel for schedule(dynamic, 8)` on rows |
| Hough accumulator increment | `#pragma omp atomic` |

The inner pixel loops run within their assigned rows. The loop-local variables are private to each iteration; threads write distinct output pixels. The Hough accumulator is shared, so its increments need atomic protection. The recently changed Hough schedule hands out groups of eight rows as threads finish; rows can have different numbers of evidence points. The other row loops keep static scheduling.

`OMP_NUM_THREADS` controls the number of OpenMP threads, for example `OMP_NUM_THREADS=4 ros2 run pcap camera_test`. The ROS package builds the OpenMP C implementation. For offline comparisons, `bench/benchmark_native.py` compiles that same source with and without `-fopenmp`, checks output equality on saved frames, and times grayscale-only and full-pipeline calls. See `bench/RESULTS.md` for the measured setup and results. There are no separately exposed timings for every internal C stage yet. Display time and Gazebo delivery rate are separate from C processing time.

## Build, run, and test

Build from the ROS workspace root, **outside `src`**:

```bash
cd /home/aryan/ros2_pcap
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select pcap
source install/setup.bash
```

Start the simulation and perception node in separate sourced terminals:

```bash
ros2 launch pcap bot.launch.py
```

```bash
ros2 run pcap camera_test
# Or, without the Matplotlib window:
ros2 run pcap camera_test --ros-args -p show_window:=false
```

`setup.py` installs the URDF, meshes, world, launch file, and model assets, builds `pcap.libperception` from `perception.c`, and registers `camera_test` as a console command. `package.xml` declares the ROS and Python runtime dependencies. The C extension is a shared library beside the Python package; `camera_test.py` locates its platform-specific filename and loads it with `ctypes.CDLL`.

The focused native tests can be run from this repository root:

```bash
OMP_NUM_THREADS=4 python3 -m pytest -q pcap/test/test_perception.py
```

Run the offline serial/OpenMP benchmark from this repository root with:

```bash
python3 bench/benchmark_native.py --repetitions 80 --trials 5 --warmup 10 --cpus 0 1 2 3 4 5 6 7
```

The test fixture compiles `perception.c` into a temporary shared library with GCC, `-O3`, and OpenMP. Tests cover blank, left-only, right-only, paired, narrow, horizontal-stripe, slanted, tiny, and invalid-dimension cases, plus two saved Gazebo frames. The third saved frame, `gazebo_camera_rgb8_failing_002.npy`, is available for diagnosis but is not currently an assertion in that test file. The tests exercise native processing without ROS or Gazebo running. They do not measure perception accuracy against labelled ground truth.

## Current limits and useful debugging checks

- The detector models each side as a straight line. Curved roads can only be approximated by that model.
- The color mask is necessary to the current Hough voting path. A strong Sobel edge can be ignored when the road-color mask merges both sides of it, as seen on the saved pale-gravel failure frame.
- The fixed ROI excludes the top quarter and bottom part of the image. It can miss a boundary that appears only outside that range; it does not adapt to camera pitch or a measured horizon.
- The mask and thresholds were designed around synthetic Gazebo imagery. There is no labelled success-rate, false-positive, IoU, or confidence-calibration result in the repository.
- There is no temporal tracking, persistence, motion compensation, or autonomous steering in `camera_test.py`.

When a result looks wrong, inspect the panels in order: confirm the road is present in **Original RGB**; check whether the apparent edge survives **Blurred** and appears in **Edges**; then check whether any line appears in **Hough Lines** and whether a pair survives in **Boundary Mask**. Remember that the Edges panel combines Sobel pixels and sparse color-boundary points, while Hough votes only on the latter. An empty **Traversable Mask** is expected whenever the validity state is not 3.
