"""Tests for ROS camera-message validation and row conversion."""

from types import SimpleNamespace

from fcv.camera_test import create_display_frame
from fcv.camera_test import rgb8_message_to_array
import numpy as np
import pytest


def _message(data, width=2, height=2, step=8, encoding='rgb8'):
    return SimpleNamespace(
        data=data,
        width=width,
        height=height,
        step=step,
        encoding=encoding,
    )


def test_rgb8_message_removes_row_padding():
    data = bytes([
        1, 2, 3, 4, 5, 6, 99, 99,
        7, 8, 9, 10, 11, 12, 88, 88,
    ])
    image = rgb8_message_to_array(_message(data))
    expected = np.array(
        [[[1, 2, 3], [4, 5, 6]],
         [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8,
    )
    assert image.flags.c_contiguous
    assert np.array_equal(image, expected)


def test_rgb8_message_rejects_wrong_encoding():
    with pytest.raises(ValueError, match='Expected rgb8, got bgr8'):
        rgb8_message_to_array(_message(bytes(16), encoding='bgr8'))


@pytest.mark.parametrize('message', [
    _message(bytes(16), width=0),
    _message(bytes(16), height=0),
    _message(bytes(12), step=5),
    _message(bytes(15)),
])
def test_rgb8_message_rejects_invalid_dimensions(message):
    with pytest.raises(ValueError, match='Invalid RGB image buffer dimensions'):
        rgb8_message_to_array(message)


def test_display_frame_has_two_rows_of_three_half_size_panels():
    perception = SimpleNamespace(validity=3, confidence=0.75)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    grayscale = np.zeros((480, 640), dtype=np.uint8)

    display = create_display_frame(
        perception,
        (rgb, grayscale, grayscale, grayscale, grayscale, grayscale),
    )

    assert display.shape == (564, 960, 3)
    assert display.dtype == np.uint8
