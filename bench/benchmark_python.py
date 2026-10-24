"""Benchmark the serial Python perception pipeline on saved frames."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import cv2
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / 'fcv'))

from fcv.perception import process_traversability_pipeline  # noqa: E402


FRAMES = REPOSITORY / 'fcv/test/frames'


def _statistics(samples_ns):
    samples_ms = [sample / 1e6 for sample in samples_ns]
    ordered = sorted(samples_ms)
    return {
        'mean_ms': statistics.mean(samples_ms),
        'median_ms': statistics.median(samples_ms),
        'p90_ms': ordered[int(0.9 * (len(ordered) - 1))],
        'min_ms': ordered[0],
        'max_ms': ordered[-1],
    }


def _load_frames():
    frames = []
    for path in sorted(FRAMES.glob('*.npy')):
        image = np.load(path, allow_pickle=False)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'{path} must contain an H x W x 3 RGB8 image')
        frames.append((path, np.ascontiguousarray(image)))
    if not frames:
        raise RuntimeError(f'No saved frames found in {FRAMES}')
    return frames


def _output_digest(result, outputs):
    digest = hashlib.sha256()
    for output in (
        outputs.gray,
        outputs.denoised,
        outputs.edges,
        outputs.road_mask,
        outputs.hough_lines,
        outputs.boundary_mask,
        outputs.traversable_mask,
    ):
        digest.update(output.tobytes())
    digest.update(
        repr((
            result.validity,
            result.confidence,
            result.left_confidence,
            result.right_confidence,
        )).encode()
    )
    return digest.hexdigest()


def _write_csv(path, frame_results):
    columns = [
        'frame', 'samples', 'mean_ms', 'median_ms', 'p90_ms',
        'min_ms', 'max_ms', 'validity', 'output_sha256',
    ]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in frame_results:
            row = {
                'frame': result['frame'],
                'samples': len(result['samples_ns']),
                'validity': result['validity'],
                'output_sha256': result['output_sha256'],
            }
            row.update({
                key: f'{value:.6f}'
                for key, value in _statistics(result['samples_ns']).items()
            })
            writer.writerow(row)


def benchmark(args):
    """Run the benchmark and write JSON and CSV reports."""
    frame_results = []
    for path, image in _load_frames():
        for _ in range(args.warmup):
            process_traversability_pipeline(image)

        samples_ns = []
        result = None
        outputs = None
        for _ in range(args.repetitions):
            start = time.perf_counter_ns()
            result, outputs = process_traversability_pipeline(image)
            samples_ns.append(time.perf_counter_ns() - start)

        frame_result = {
            'frame': path.name,
            'shape': list(image.shape),
            'validity': result.validity,
            'output_sha256': _output_digest(result, outputs),
            'samples_ns': samples_ns,
        }
        frame_results.append(frame_result)
        print(
            f'{path.name}: '
            f'{_statistics(samples_ns)["median_ms"]:.3f} ms median'
        )

    output = Path(args.output).resolve() if args.output else (
        REPOSITORY / 'bench/results'
        / datetime.now(timezone.utc).strftime('python_%Y%m%d_%H%M%S.json')
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'description': (
            'Serial Python pipeline timing; excludes ROS, Gazebo, plotting, '
            'and frame loading; includes output allocation'
        ),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'opencv': cv2.__version__,
        'configuration': {
            'warmup': args.warmup,
            'repetitions': args.repetitions,
        },
        'frames': frame_results,
    }
    output.write_text(json.dumps(report, indent=2) + '\n')
    csv_path = output.with_suffix('.csv')
    _write_csv(csv_path, frame_results)
    print(f'JSON: {output}')
    print(f'CSV: {csv_path}')


def main():
    """Parse command-line arguments and run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--output', help='JSON output path; CSV uses same stem')
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions < 2:
        parser.error('Need nonnegative warmup and at least two repetitions')
    benchmark(args)


if __name__ == '__main__':
    main()
