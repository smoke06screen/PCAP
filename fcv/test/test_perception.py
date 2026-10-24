"""Synthetic checks for boundary selection and safe mask behavior."""

import hashlib
from pathlib import Path

from fcv.perception import process_traversability_pipeline
import numpy as np
import pytest


GOLDEN_FRAMES = {
    'gazebo_camera_rgb8_001.npy': (
        '1130af324ebb17b5004fc9870d49801fa745c69fe7f2822c6261db3a74d90afc',
        (3, 0.7106888890266418, 0.6201391816139221,
         0.6879087090492249),
    ),
    'gazebo_camera_rgb8_failing_001.npy': (
        '967ee7faeeae83e9cca149eecc6d65a8d13b006304f02f1f125c45ab034ac5d2',
        (3, 0.6855838894844055, 0.615370512008667,
         0.6590641736984253),
    ),
    'gazebo_camera_rgb8_failing_002.npy': (
        'dde00554a4c62c9262046d2bd601f0f650e9c9eb04b992f0d7c5682b20e9c245',
        (1, 0.7189885377883911, 0.7189885377883911, 0.0),
    ),
}


def run_scene(image):
    result, outputs = process_traversability_pipeline(image)
    return result, [
        outputs.gray,
        outputs.denoised,
        outputs.edges,
        outputs.road_mask,
        outputs.hough_lines,
        outputs.boundary_mask,
        outputs.traversable_mask,
    ]


@pytest.mark.parametrize('filename', GOLDEN_FRAMES)
def test_saved_frame_exact_regression(filename):
    expected_digest, expected_result = GOLDEN_FRAMES[filename]
    frame = np.load(Path(__file__).parent / 'frames' / filename)
    result, outputs = run_scene(frame)
    digest = hashlib.sha256()
    for output in outputs:
        digest.update(output.tobytes())

    assert digest.hexdigest() == expected_digest
    assert (
        result.validity,
        result.confidence,
        result.left_confidence,
        result.right_confidence,
    ) == expected_result
    if filename.endswith('failing_002.npy'):
        assert result.validity == 1
        assert not np.any(outputs[-1])


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
    elif kind == 'slanted_pair':
        for y in range(image.shape[0]):
            left = round(65 - (y - 36) * 20 / 56)
            right = round(95 + (y - 36) * 20 / 56)
            image[y, left:right] = 200
    return image


@pytest.mark.parametrize('kind,validity', [
    ('blank', 0), ('pair', 3), ('left', 1), ('right', 2),
    ('narrow', 4), ('horizontal', 0),
])
def test_boundary_states(kind, validity):
    result, outputs = run_scene(scene(kind))
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


def test_invalid_dimensions_and_type():
    with pytest.raises(ValueError):
        process_traversability_pipeline(np.empty((0, 160, 3), np.uint8))
    with pytest.raises(ValueError):
        process_traversability_pipeline(np.empty((120, 160, 3), np.float32))


def test_horizontal_stripe_does_not_replace_boundary_pair():
    result, outputs = run_scene(scene('pair_with_stripe'))
    assert result.validity == 3
    assert outputs[-1][60, 80] == 255
    assert outputs[-1][68, 80] == 0
    assert outputs[-1][75, 80] == 255


def test_tiny_frame_has_empty_safe_masks():
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    result, outputs = run_scene(image)
    assert result.validity == 0
    assert not np.any(outputs[-1])
    assert not np.any(outputs[-2])


def test_moderate_inward_slant_scores_above_vertical():
    slanted, outputs = run_scene(scene('slanted_pair'))
    vertical, _ = run_scene(scene('pair'))
    assert slanted.validity == 3
    assert outputs[-1][60, 80] == 255
    assert slanted.left_confidence > vertical.left_confidence
    assert slanted.right_confidence > vertical.right_confidence


def test_saved_gazebo_frame_with_clipped_boundaries():
    frame = np.load(Path(__file__).parent / 'frames/gazebo_camera_rgb8_001.npy')
    result, outputs = run_scene(frame)
    assert result.validity == 3
    boundary = outputs[-2]
    for y, expected_left, expected_right in [(144, 198, 482),
                                             (220, 92, 616)]:
        points = np.flatnonzero(boundary[y])
        assert abs(points[0] - expected_left) <= 12
        assert abs(points[-1] - expected_right) <= 12
    assert outputs[-1][200, 320] == 255
    assert outputs[-1][350, 320] == 0


def test_saved_short_left_boundary_below_world_horizon():
    frame = np.load(Path(__file__).parent /
                    'frames/gazebo_camera_rgb8_failing_001.npy')
    result, outputs = run_scene(frame)
    assert result.validity == 3
    boundary = outputs[-2]
    for y, expected_left, expected_right in [(144, 114, 401),
                                             (200, 21, 480)]:
        points = np.flatnonzero(boundary[y])
        assert abs(points[0] - expected_left) <= 15
        assert abs(points[-1] - expected_right) <= 15
    assert not np.any(boundary[70])
    assert outputs[-1][180, 300] == 255
    assert outputs[-1][180, 20] == 0
