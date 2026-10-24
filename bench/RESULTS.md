# Serial Python benchmark

The current benchmark times `process_traversability_pipeline()` on the three
saved 640×480 Gazebo RGB frames. It excludes frame loading, ROS, Gazebo, and
Matplotlib, but includes creation of the result arrays.

Run it from the repository root:

```bash
python3 bench/benchmark_python.py --repetitions 20 --warmup 3
```

It writes JSON with every timing sample and a CSV summary containing mean,
median, 90th percentile, minimum, and maximum time for each frame. It also
hashes every output so repeated runs can be checked for identical results.

The tracked `results/native_*` files are historical measurements from the
previous C/OpenMP version. They are retained as old experimental data only and
must not be reported as performance results for the current Python branch.
