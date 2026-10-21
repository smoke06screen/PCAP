# Serial C versus OpenMP measurements

These are measurements from the current native C pipeline on the three saved 640 × 480 Gazebo RGB frames. They are **offline native-call timings**: the timer surrounds one Python `ctypes` call into C. Frame loading, array allocation, ROS message delivery, Gazebo, and Matplotlib are outside the timer. The small Python-to-C dispatch cost is inside it.

## How to reproduce

From this repository root:

```bash
python3 bench/benchmark_native.py --repetitions 80 --trials 5 --warmup 10 --cpus 0 1 2 3 4 5 6 7
```

The script compiles `pcap/pcap/native/perception.c` twice at `-O3`: once without `-fopenmp` for serial C, and once with `-fopenmp` for OpenMP C. The source, algorithms, and frame inputs are otherwise identical. The serial compiler ignores the OpenMP pragmas. Each run uses 10 warmup calls and 80 timed calls per frame; five randomized trial blocks give 400 timed calls for each frame/condition. Parallel runs use `OMP_NUM_THREADS` values 1, 2, 4, and 8; `OMP_DYNAMIC=FALSE`, `OMP_PROC_BIND=spread`, and `OMP_PLACES=threads`. Every worker process in the reported run was restricted to logical CPUs 0–7, which are the eight hardware threads of four performance cores on this machine.

The benchmark hashes every output buffer and the result structure after each run. All serial and OpenMP hashes matched across all frames, thread counts, and trials. This checks computational equivalence; it is not a measure of whether the road boundaries are visually correct.

## Environment and results

- Machine: Intel Core Ultra 5 125H, 18 logical CPUs total; the reported run used CPUs 0–7.
- OS: Linux 6.8.0-138-generic, x86-64.
- Compiler: GCC 11.4.0, `-std=c11 -O3 -fPIC -shared`; OpenMP build adds `-fopenmp`; both link `-lm`.
- Python/NumPy: Python 3.10.12, NumPy 1.21.5. Exact versions, source/frame SHA-256 hashes, build commands, and all sample timings are in the JSON file below.
- Inputs: `gazebo_camera_rgb8_001.npy`, `gazebo_camera_rgb8_failing_001.npy`, and `gazebo_camera_rgb8_failing_002.npy`.

The table reports the **median of the three frame-specific medians**. Each frame-specific median comes from 400 calls. Speedup is the serial value divided by the OpenMP value. Efficiency is speedup divided by the OpenMP thread count.

| Stage | Build | Threads | Time (ms) | Speedup | Efficiency |
| --- | --- | ---: | ---: | ---: | ---: |
| Full pipeline | Serial C | 1 | 4.979 | 1.00× | — |
| Full pipeline | OpenMP C | 1 | 5.673 | 0.88× | 0.88 |
| Full pipeline | OpenMP C | 2 | 3.135 | 1.59× | 0.79 |
| Full pipeline | OpenMP C | 4 | 2.333 | 2.13× | 0.53 |
| Full pipeline | OpenMP C | 8 | 2.133 | 2.33× | 0.29 |
| Grayscale entry point | Serial C | 1 | 0.654 | 1.00× | — |
| Grayscale entry point | OpenMP C | 1 | 0.853 | 0.77× | 0.77 |
| Grayscale entry point | OpenMP C | 2 | 0.469 | 1.39× | 0.70 |
| Grayscale entry point | OpenMP C | 4 | 0.340 | 1.92× | 0.48 |
| Grayscale entry point | OpenMP C | 8 | 0.206 | 3.17× | 0.40 |

The grayscale entry point calls the C grayscale wrapper and does not include the later pipeline stages. Full-pipeline timing includes C allocation/free and all processing stages, including Hough voting and mask generation. OpenMP with one thread is slower than serial C here because it still enters parallel regions. The eight-thread full-pipeline result improves on four threads only modestly; overhead, atomic Hough votes, and sharing four physical cores among eight hardware threads limit scaling. These are interpretations of the measured behavior, not isolated stage-level timing results.

Some trials contained large latency outliers and the two- and four-thread medians varied between trial blocks. Use the per-frame median, mean, 90th percentile, and raw samples in the artifacts when discussing variability. For a publication-ready final result, rerun on an otherwise idle machine with the CPU placement, compiler, frame set, and environment recorded. Do not use the camera's requested 10 Hz update rate as native processing throughput.

## Saved data

- Primary pinned run: [`results/native_20261021_164042.json`](results/native_20261021_164042.json) and [`results/native_20261021_164042.csv`](results/native_20261021_164042.csv).
- Earlier unpinned comparison: [`results/native_20261021_163856.json`](results/native_20261021_163856.json) and [`results/native_20261021_163856.csv`](results/native_20261021_163856.csv). This run used all CPUs available to the process and is not the source of the main table.

The CSV gives each frame's mean, median, 90th percentile, min/max, speedup, and efficiency. The JSON also contains every individual timing sample and the exact source and input hashes.

## Follow-up: all 18 logical CPUs

The first table deliberately restricted execution to CPUs 0–7, which are four physical performance cores with two hardware threads each. This keeps that comparison within one core type. The machine also has efficiency cores, so a separate run allowed all logical CPUs 0–17 and tested up to 18 OpenMP threads:

```bash
python3 bench/benchmark_native.py --repetitions 80 --trials 5 --warmup 10 \
  --threads 1 2 4 8 14 18 --cpus 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
```

The build, inputs, warmups, trials, and timing method were the same. Every serial/OpenMP output hash matched. The figures below again show the median of three frame-specific medians, each based on 400 calls. **Do not combine these timings with the restricted-CPU table above:** the available cores and thread placement differ.

| Build | Threads | Full pipeline (ms) | Grayscale entry point (ms) |
| --- | ---: | ---: | ---: |
| Serial C | 1 | 4.892 | 0.724 |
| OpenMP C | 1 | 5.988 | 0.817 |
| OpenMP C | 2 | 5.338 | 0.600 |
| OpenMP C | 4 | 3.236 | 0.351 |
| OpenMP C | 8 | 2.386 | 0.232 |
| OpenMP C | 14 | 2.487 | 0.127 |
| OpenMP C | 18 | 52.087 | 0.151 |

On this run, eight threads gave the lowest overall-pipeline median. The 18-thread pipeline was much slower despite the extra CPU threads, while grayscale remained much faster than serial. That points to costs in later stages or their interaction with mixed-core scheduling; the current benchmark does not isolate which stage caused the 18-thread slowdown. A future stage-level profile should test that specifically. This is one machine and workload, and the raw samples show substantial variation.

Whole-machine data: [`results/native_20261021_164554.json`](results/native_20261021_164554.json) and [`results/native_20261021_164554.csv`](results/native_20261021_164554.csv).
