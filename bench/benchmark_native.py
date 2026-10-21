"""Compare the same native perception source built with and without OpenMP.

Run from any directory with:
    python3 bench/benchmark_native.py --repetitions 80 --trials 5

Timing surrounds one ctypes call into C. It excludes loading frames, allocation,
ROS, Gazebo, and plotting, but includes the small Python-to-C call overhead.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time

import ctypes
import numpy as np


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'pcap/pcap/native/perception.c'
FRAMES = REPO / 'pcap/test/frames'
STAGES = ('grayscale', 'full_pipeline')


class PerceptionResult(ctypes.Structure):
    _fields_ = [
        ('validity', ctypes.c_int),
        ('confidence', ctypes.c_float),
        ('left_confidence', ctypes.c_float),
        ('right_confidence', ctypes.c_float),
    ]


def build_libraries(directory):
    common = ['gcc', '-std=c11', '-O3', '-fPIC', '-shared', str(SOURCE)]
    commands = {
        'serial': common + ['-lm', '-o', str(directory / 'serial.so')],
        'openmp': common + ['-fopenmp', '-lm', '-o', str(directory / 'openmp.so')],
    }
    for command in commands.values():
        subprocess.run(command, check=True, capture_output=True, text=True)
    return commands


def load_frames(paths):
    frames = []
    for path in paths:
        image = np.load(path, allow_pickle=False)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'{path} must contain an H x W x 3 uint8 RGB image')
        image = np.ascontiguousarray(image)
        frames.append((path, image))
    return frames


def worker(args):
    if args.cpus:
        os.sched_setaffinity(0, set(args.cpus))
    library = ctypes.CDLL(args.library)
    byte_ptr = ctypes.POINTER(ctypes.c_uint8)
    library.process_rgb8_to_gray.argtypes = [byte_ptr, byte_ptr, ctypes.c_int, ctypes.c_int]
    library.process_rgb8_to_gray.restype = ctypes.c_int
    library.process_traversability_pipeline_ex.argtypes = (
        [byte_ptr] * 8 + [ctypes.c_int, ctypes.c_int, ctypes.POINTER(PerceptionResult)]
    )
    library.process_traversability_pipeline_ex.restype = ctypes.c_int

    frame_results = []
    for path, image in load_frames(args.frame):
        height, width = image.shape[:2]
        input_ptr = image.ctypes.data_as(byte_ptr)
        output_count = 1 if args.stage == 'grayscale' else 7
        outputs = [np.empty((height, width), dtype=np.uint8) for _ in range(output_count)]
        output_ptrs = [array.ctypes.data_as(byte_ptr) for array in outputs]
        result = PerceptionResult()

        if args.stage == 'grayscale':
            def call():
                return library.process_rgb8_to_gray(input_ptr, output_ptrs[0], width, height)
        else:
            def call():
                return library.process_traversability_pipeline_ex(
                    input_ptr, *output_ptrs, width, height, ctypes.byref(result)
                )

        for _ in range(args.warmup):
            if call() != 0:
                raise RuntimeError(f'C call failed during warmup: {path}')

        samples_ns = []
        for _ in range(args.repetitions):
            start = time.perf_counter_ns()
            status = call()
            elapsed = time.perf_counter_ns() - start
            if status != 0:
                raise RuntimeError(f'C call failed during measurement: {path}')
            samples_ns.append(elapsed)

        digest = hashlib.sha256()
        for output in outputs:
            digest.update(output.tobytes())
        if args.stage == 'full_pipeline':
            digest.update(bytes(result))
        frame_results.append({
            'frame': path.name,
            'shape': list(image.shape),
            'output_sha256': digest.hexdigest(),
            'validity': result.validity if args.stage == 'full_pipeline' else None,
            'samples_ns': samples_ns,
        })
    return frame_results


def cpu_description():
    result = subprocess.run(['lscpu'], capture_output=True, text=True, check=True)
    keys = ('Model name', 'CPU(s)', 'Core(s) per socket', 'Thread(s) per core')
    values = {}
    for line in result.stdout.splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            if key.strip() in keys:
                values[key.strip()] = value.strip()
    return values


def describe(samples_ns):
    samples_ms = [value / 1e6 for value in samples_ns]
    ordered = sorted(samples_ms)
    return {
        'mean_ms': statistics.mean(samples_ms),
        'median_ms': statistics.median(samples_ms),
        'p90_ms': ordered[int(0.9 * (len(ordered) - 1))],
        'min_ms': ordered[0],
        'max_ms': ordered[-1],
    }


def write_csv(path, runs):
    grouped = {}
    for run in runs:
        for frame in run['frames']:
            key = (run['stage'], frame['frame'], run['variant'], run['threads'])
            grouped.setdefault(key, []).extend(frame['samples_ns'])
    serial_medians = {
        (stage, frame): describe(samples)['median_ms']
        for (stage, frame, variant, _), samples in grouped.items()
        if variant == 'serial'
    }
    columns = ['stage', 'frame', 'variant', 'threads', 'samples',
               'mean_ms', 'median_ms', 'p90_ms', 'min_ms', 'max_ms',
               'speedup_vs_serial', 'parallel_efficiency']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for key, samples in sorted(grouped.items()):
            stage, frame, variant, count = key
            stats = describe(samples)
            speedup = serial_medians[(stage, frame)] / stats['median_ms']
            writer.writerow({
                'stage': stage, 'frame': frame, 'variant': variant,
                'threads': count, 'samples': len(samples),
                **{name: f'{value:.6f}' for name, value in stats.items()},
                'speedup_vs_serial': f'{speedup:.6f}',
                'parallel_efficiency': f'{speedup / count:.6f}' if variant == 'openmp' else '',
            })


def benchmark(args):
    paths = sorted(FRAMES.glob('*.npy'))
    if not paths:
        raise RuntimeError(f'No saved frames found in {FRAMES}')
    load_frames(paths)
    output = Path(args.output).resolve() if args.output else (
        REPO / 'bench/results' / datetime.now(timezone.utc).strftime('native_%Y%m%d_%H%M%S.json')
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    with tempfile.TemporaryDirectory(prefix='pcap_native_benchmark_') as temp:
        commands = build_libraries(Path(temp))
        runs = []
        expected = {}
        for trial in range(args.trials):
            cases = [(stage, 'serial', 1) for stage in STAGES]
            cases += [(stage, 'openmp', count) for stage in STAGES for count in args.threads]
            rng.shuffle(cases)
            for stage, variant, count in cases:
                env = dict(os.environ, OMP_NUM_THREADS=str(count), OMP_DYNAMIC='FALSE',
                           OMP_PROC_BIND='spread', OMP_PLACES='threads')
                command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                           '--library', str(Path(temp) / f'{variant}.so'),
                           '--stage', stage, '--warmup', str(args.warmup),
                           '--repetitions', str(args.repetitions)]
                if args.cpus:
                    command += ['--cpus', *(str(cpu) for cpu in args.cpus)]
                for path in paths:
                    command += ['--frame', str(path)]
                process = subprocess.run(command, check=True, capture_output=True,
                                         text=True, env=env)
                frames = json.loads(process.stdout)
                for frame in frames:
                    key = (stage, frame['frame'])
                    if key in expected and frame['output_sha256'] != expected[key]:
                        raise RuntimeError(f'Output mismatch for {stage}, {frame["frame"]}, '
                                           f'{variant}, {count} threads')
                    expected[key] = frame['output_sha256']
                runs.append({'trial': trial, 'stage': stage, 'variant': variant,
                             'threads': count, 'frames': frames})
                medians = [round(describe(frame['samples_ns'])['median_ms'], 3)
                           for frame in frames]
                print(f'{trial + 1}/{args.trials} {stage} {variant} '
                      f'{count} thread(s): {medians} ms', flush=True)

        report = {
            'description': 'Native C call timing; excludes ROS, Gazebo, plotting, '
                           'frame loading, and output allocation; includes ctypes dispatch',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'source': str(SOURCE.relative_to(REPO)),
            'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'frames': [{'name': path.name,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                       for path in paths],
            'cpu': cpu_description(),
            'platform': platform.platform(),
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'compiler': subprocess.check_output(['gcc', '--version'], text=True).splitlines()[0],
            'compile_commands': commands,
            'openmp_environment': {'OMP_DYNAMIC': 'FALSE', 'OMP_PROC_BIND': 'spread',
                                   'OMP_PLACES': 'threads'},
            'configuration': {'threads': args.threads, 'warmup': args.warmup,
                              'repetitions': args.repetitions, 'trials': args.trials,
                              'seed': args.seed, 'cpus': args.cpus},
            'output_equality': 'SHA-256 of all output arrays and result structure matched '
                               'for every serial/OpenMP run',
            'runs': runs,
        }
        output.write_text(json.dumps(report, indent=2) + '\n')
        csv_path = output.with_suffix('.csv')
        write_csv(csv_path, runs)
        print(f'JSON: {output}')
        print(f'CSV: {csv_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--library', help=argparse.SUPPRESS)
    parser.add_argument('--stage', choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument('--frame', type=Path, action='append', help=argparse.SUPPRESS)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--cpus', type=int, nargs='+',
                        help='Allowed logical CPU IDs for every worker process')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repetitions', type=int, default=80)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output', help='Output JSON path; CSV uses the same stem')
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
    else:
        if args.repetitions < 2 or args.trials < 1 or args.warmup < 0:
            parser.error('Need at least two repetitions, one trial, and nonnegative warmup')
        if not args.threads or any(count < 1 for count in args.threads):
            parser.error('Thread counts must be positive')
        if args.cpus and (any(cpu < 0 for cpu in args.cpus) or
                          not set(args.cpus).issubset(os.sched_getaffinity(0))):
            parser.error('CPU IDs must belong to the current process affinity')
        benchmark(args)


if __name__ == '__main__':
    main()
