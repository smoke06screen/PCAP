"""Synthetic checks for native boundary selection and safe mask behavior."""

import ctypes
from pathlib import Path
import subprocess

import numpy as np
import pytest


class PerceptionResult(ctypes.Structure):
    _fields_ = [
        ('validity', ctypes.c_int),
        ('confidence', ctypes.c_float),
        ('left_confidence', ctypes.c_float),
        ('right_confidence', ctypes.c_float),
    ]


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    source = Path(__file__).resolve().parents[1] / 'pcap/native/perception.c'
    library = tmp_path_factory.mktemp('native') / 'libperception.so'
    subprocess.run(
        ['gcc', '-std=c11', '-O3', '-fopenmp', '-Wall', '-Wextra',
         '-Wpedantic', '-fPIC', '-shared', str(source), '-lm', '-o', str(library)],
        check=True,
    )
    lib = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_uint8)
    lib.process_traversability_pipeline_ex.argtypes = (
        [pointer] * 8 + [ctypes.c_int, ctypes.c_int,
                         ctypes.POINTER(PerceptionResult)]
    )
    lib.process_traversability_pipeline_ex.restype = ctypes.c_int
    lib.process_traversability_pipeline.argtypes = (
        [pointer] * 8 + [ctypes.c_int, ctypes.c_int]
    )
    lib.process_traversability_pipeline.restype = ctypes.c_int
    return lib


def run_scene(lib, scene):
    height, width = scene.shape[:2]
    outputs = [np.empty((height, width), dtype=np.uint8) for _ in range(7)]
    pointer = ctypes.POINTER(ctypes.c_uint8)
    def address(array):
        return array.ctypes.data_as(pointer)
    result = PerceptionResult()
    status = lib.process_traversability_pipeline_ex(
        address(scene), *(address(output) for output in outputs),
        width, height, ctypes.byref(result),
    )
    assert status == 0
    return result, outputs


def scene(kind):
    image = np.empty((120, 160, 3), dtype=np.uint8)
    image[:] = (0, 150, 0)
    if kind == 'pair':
        image[:, 50:110] = 200
    elif kind == 'left':
        image[:, 50:] = 200
    elif kind == 'right':
        image[:, :110] = 200
    elif kind == 'narrow':
        image[:, 72:88] = 200
    elif kind == 'horizontal':
        image[70:74, :] = 200
    elif kind == 'pair_with_stripe':
        image[:, 50:110] = 200
        image[66:71, :] = (0, 150, 0)
    return image


@pytest.mark.parametrize('kind,validity', [
    ('blank', 0), ('pair', 3), ('left', 1), ('right', 2),
    ('narrow', 4), ('horizontal', 0),
])
def test_boundary_states(native, kind, validity):
    result, outputs = run_scene(native, scene(kind))
    assert result.validity == validity
    assert 0.0 <= result.confidence <= 1.0
    traversable = outputs[-1]
    if kind == 'pair':
        assert traversable[60, 80] == 255
        assert traversable[60, 20] == 0
        assert np.count_nonzero(traversable) > 1000
        assert np.count_nonzero(outputs[-2]) > 0
        assert result.left_confidence > 0.5
        assert result.right_confidence > 0.5
    else:
        assert not np.any(traversable)


def test_legacy_entry_point_and_invalid_dimensions(native):
    image = scene('pair')
    outputs = [np.empty((120, 160), dtype=np.uint8) for _ in range(7)]
    pointer = ctypes.POINTER(ctypes.c_uint8)
    def address(array):
        return array.ctypes.data_as(pointer)
    status = native.process_traversability_pipeline(
        address(image), *(address(output) for output in outputs), 160, 120,
    )
    assert status == 0
    assert outputs[-1][60, 80] == 255
    result = PerceptionResult()
    assert native.process_traversability_pipeline_ex(
        address(image), *(address(output) for output in outputs),
        0, 120, ctypes.byref(result),
    ) == -1


def test_horizontal_stripe_does_not_replace_boundary_pair(native):
    result, outputs = run_scene(native, scene('pair_with_stripe'))
    assert result.validity == 3
    assert outputs[-1][60, 80] == 255
    assert outputs[-1][68, 80] == 0
    assert outputs[-1][75, 80] == 255


def test_tiny_frame_has_empty_safe_masks(native):
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    result, outputs = run_scene(native, image)
    assert result.validity == 0
    assert not np.any(outputs[-1])
    assert not np.any(outputs[-2])
