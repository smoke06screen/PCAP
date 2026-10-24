"""Serial Python implementation of the FCV traversability pipeline."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


# Keep OpenCV kernels serial as required by this implementation.
cv2.setNumThreads(1)


NO_BOUNDARY = 0
LEFT_ONLY = 1
RIGHT_ONLY = 2
VALID_PAIR = 3
PAIR_REJECTED = 4
LOW_CONFIDENCE = 5

THETA_BINS = 180
MAX_CANDIDATES = 24
SUPPORT_RADIUS = 2


@dataclass(frozen=True)
class PerceptionResult:
    """Boundary validity and confidence scores for one image."""

    validity: int = NO_BOUNDARY
    confidence: float = 0.0
    left_confidence: float = 0.0
    right_confidence: float = 0.0


@dataclass(frozen=True)
class PipelineOutputs:
    """Intermediate and final single-channel pipeline images."""

    gray: np.ndarray
    denoised: np.ndarray
    edges: np.ndarray
    road_mask: np.ndarray
    hough_lines: np.ndarray
    boundary_mask: np.ndarray
    traversable_mask: np.ndarray


@dataclass
class _Candidate:
    theta: int
    rho: int
    votes: int
    x_top: float
    x_bottom: float
    orientation: float = 0.0
    support: float = 0.0
    score: float = 0.0


def _vehicle_mask(width, height):
    mask = np.zeros((height, width), dtype=bool)
    vehicle_top = height * 78 // 100
    vehicle_bottom = height - 1
    center = np.float32(width) * np.float32(0.5)
    top_half_width = np.float32(width) * np.float32(0.12)
    bottom_half_width = np.float32(width) * np.float32(0.21)

    for y in range(vehicle_top, height):
        if vehicle_bottom > vehicle_top:
            ratio = np.float32(y - vehicle_top) / np.float32(
                vehicle_bottom - vehicle_top
            )
        else:
            ratio = np.float32(1.0)
        half_width = top_half_width + (
            bottom_half_width - top_half_width
        ) * ratio
        left = int(center - half_width)
        right = int(center + half_width)
        mask[y, max(0, left):min(width, right + 1)] = True

    return mask


def rgb8_to_gray(rgb):
    """Convert a contiguous RGB8 image with the original coefficients."""
    _validate_rgb8(rgb)
    rgb_float = rgb.astype(np.float32)
    luminance = (
        np.float32(0.299) * rgb_float[:, :, 0]
        + np.float32(0.587) * rgb_float[:, :, 1]
        + np.float32(0.114) * rgb_float[:, :, 2]
    )
    return (luminance + np.float32(0.5)).astype(np.uint8)


def _validate_rgb8(rgb):
    if not isinstance(rgb, np.ndarray):
        raise TypeError('rgb must be a NumPy array')
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('rgb must have dtype uint8 and shape (height, width, 3)')
    if rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
        raise ValueError('rgb dimensions must be positive')


def _sobel_edges(image):
    gx = cv2.Sobel(
        image, cv2.CV_16S, 1, 0, ksize=3,
        borderType=cv2.BORDER_CONSTANT,
    )
    gy = cv2.Sobel(
        image, cv2.CV_16S, 0, 1, ksize=3,
        borderType=cv2.BORDER_CONSTANT,
    )
    magnitude = np.abs(gx.astype(np.int32)) + np.abs(gy.astype(np.int32))
    edges = np.where(magnitude > 120, 255, 0).astype(np.uint8)
    edges[0, :] = 0
    edges[-1, :] = 0
    edges[:, 0] = 0
    edges[:, -1] = 0
    return edges


def _road_candidate_mask(rgb, gray, vehicle_mask):
    channels = rgb.astype(np.int16)
    red = channels[:, :, 0]
    green = channels[:, :, 1]
    blue = channels[:, :, 2]
    saturation = channels.max(axis=2) - channels.min(axis=2)
    green_excess = green - ((red + blue) // 2)
    road = (
        (gray > 35)
        & (gray < 215)
        & (saturation < 90)
        & (green_excess < 18)
        & ~vehicle_mask
    )
    return road.astype(np.uint8) * 255


def _color_boundary_evidence(road_mask, debug_edges):
    height, width = road_mask.shape
    evidence = np.zeros_like(road_mask)
    top = height * 25 // 100
    bottom = height * 78 // 100 - 1
    center = width // 2

    if bottom < top:
        return evidence

    rows = road_mask[top:bottom + 1] != 0
    row_count = rows.shape[0]
    left = np.zeros(row_count, dtype=np.intp)
    right = np.full(row_count, width - 1, dtype=np.intp)

    if center > 0:
        blocked_left = ~rows[:, :center]
        has_blocked_left = np.any(blocked_left, axis=1)
        distance = np.argmax(blocked_left[:, ::-1], axis=1)
        left[has_blocked_left] = center - distance[has_blocked_left]

    if center + 1 < width:
        blocked_right = ~rows[:, center + 1:]
        has_blocked_right = np.any(blocked_right, axis=1)
        distance = np.argmax(blocked_right, axis=1)
        right[has_blocked_right] = center + distance[has_blocked_right]

    accepted = rows[:, center] & (right - left >= width // 12)
    image_rows = np.arange(top, bottom + 1)
    left_points = accepted & (left > 0)
    right_points = accepted & (right < width - 1)

    evidence[image_rows[left_points], left[left_points]] = 255
    debug_edges[image_rows[left_points], left[left_points]] = 255
    evidence[image_rows[right_points], right[right_points]] = 255
    debug_edges[image_rows[right_points], right[right_points]] = 255

    return evidence


def _line_x(rho, sine, cosine, y):
    return (rho - y * sine) / cosine


def _orientation_score(x_top, x_bottom, top, bottom, is_left):
    slope = (x_bottom - x_top) / (bottom - top)
    magnitude = abs(slope)
    if magnitude <= 0.35:
        score = 0.55 + 0.45 * magnitude / 0.35
    else:
        score = 1.0 - 0.40 * min(1.0, (magnitude - 0.35) / 1.65)
    if (is_left and slope > 0.05) or (not is_left and slope < -0.05):
        score *= 0.55
    return score


def _lround(value):
    if value >= 0.0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def _draw_candidate(mask, candidate, top, bottom):
    width = mask.shape[1]
    slope = (candidate.x_bottom - candidate.x_top) / (bottom - top)
    for y in range(top, bottom + 1):
        x = _lround(candidate.x_top + (y - top) * slope)
        if 0 <= x < width:
            mask[y, x] = 255


def _edge_support(edges, candidate, top, bottom):
    width = edges.shape[1]
    supported = 0
    visible = 0
    samples = bottom - top + 1
    slope = (candidate.x_bottom - candidate.x_top) / (bottom - top)
    for y in range(top, bottom + 1):
        x = _lround(candidate.x_top + (y - top) * slope)
        if not 0 <= x < width:
            continue
        visible += 1
        start = max(0, x - SUPPORT_RADIUS)
        end = min(width, x + SUPPORT_RADIUS + 1)
        if np.any(edges[y, start:end]):
            supported += 1
    if visible < samples // 5:
        return 0.0
    return supported / visible


def _add_candidate(candidates, candidate):
    place = len(candidates)
    if place == MAX_CANDIDATES:
        if candidate.score <= candidates[-1].score:
            return
        place -= 1
    else:
        candidates.append(candidate)

    while place > 0 and candidates[place - 1].score < candidate.score:
        if place < len(candidates):
            candidates[place] = candidates[place - 1]
        place -= 1
    candidates[place] = candidate


def _is_local_peak(accumulator, theta, rho):
    theta_bins, rho_bins = accumulator.shape
    votes = accumulator[theta, rho]
    for delta_theta in range(-2, 3):
        neighbor_theta = theta + delta_theta
        if not 0 <= neighbor_theta < theta_bins:
            continue
        for delta_rho in range(-3, 4):
            neighbor_rho = rho + delta_rho
            if (
                not 0 <= neighbor_rho < rho_bins
                or (delta_theta == 0 and delta_rho == 0)
            ):
                continue
            neighbor = accumulator[neighbor_theta, neighbor_rho]
            if neighbor > votes or (
                neighbor == votes
                and (
                    neighbor_theta < theta
                    or (neighbor_theta == theta and neighbor_rho < rho)
                )
            ):
                return False
    return True


def _pair_geometry(left, right, width, top, bottom):
    left_slope = (left.x_bottom - left.x_top) / (bottom - top)
    right_slope = (right.x_bottom - right.x_top) / (bottom - top)
    top_width = right.x_top - left.x_top
    visible_rows = 0
    last_left = 0.0
    last_right = 0.0
    for y in range(top, bottom + 1):
        left_x = left.x_top + (y - top) * left_slope
        right_x = right.x_top + (y - top) * right_slope
        if 0.0 <= left_x < width and 0.0 <= right_x < width:
            visible_rows += 1
            last_left = left_x
            last_right = right_x

    bottom_width = last_right - last_left
    if (
        visible_rows < (bottom - top + 1) // 5
        or top_width < 0.08 * width
        or bottom_width < 0.18 * width
        or top_width > 0.90 * width
        or bottom_width > 0.98 * width
        or bottom_width < 0.75 * top_width
    ):
        return -1.0

    center = (last_left + last_right) * 0.5
    center_score = 1.0 - min(1.0, abs(center / width - 0.5) / 0.5)
    convergence = min(1.0, bottom_width / top_width)
    return 0.5 * center_score + 0.5 * convergence


def _detect_boundaries(evidence, road_mask):
    height, width = road_mask.shape
    top = height * 25 // 100
    bottom = height * 78 // 100 - 1
    hough_lines = np.zeros_like(road_mask)
    boundary_mask = np.zeros_like(road_mask)
    traversable_mask = np.zeros_like(road_mask)

    if bottom <= top or width < 8:
        return PerceptionResult(), hough_lines, boundary_mask, traversable_mask

    diagonal = math.ceil(math.hypot(width, height))
    rho_bins = diagonal * 2 + 1
    cosines = [math.cos(theta * math.pi / 180.0)
               for theta in range(THETA_BINS)]
    sines = [math.sin(theta * math.pi / 180.0)
             for theta in range(THETA_BINS)]
    accumulator = np.zeros((THETA_BINS, rho_bins), dtype=np.int32)

    points = np.argwhere(evidence[top:bottom + 1] != 0)
    if points.size:
        y_values = (points[:, 0] + top).astype(np.float64)[:, None]
        x_values = points[:, 1].astype(np.float64)[:, None]
        cosine_array = np.asarray(cosines)[None, :]
        sine_array = np.asarray(sines)[None, :]
        rho_indices = (
            x_values * cosine_array
            + y_values * sine_array
            + diagonal
            + 0.5
        ).astype(np.intp)
        theta_indices = np.broadcast_to(
            np.arange(THETA_BINS), rho_indices.shape
        )
        valid = (rho_indices >= 0) & (rho_indices < rho_bins)
        np.add.at(
            accumulator,
            (theta_indices[valid], rho_indices[valid]),
            1,
        )

    left = []
    right = []
    minimum_votes = (bottom - top + 1) * 12 // 100
    for theta in range(THETA_BINS):
        if abs(cosines[theta]) < 0.35:
            continue
        possible_rhos = np.flatnonzero(
            accumulator[theta] >= minimum_votes
        )
        for rho_value in possible_rhos:
            rho_index = int(rho_value)
            votes = int(accumulator[theta, rho_index])
            if not _is_local_peak(accumulator, theta, rho_index):
                continue
            rho = rho_index - diagonal
            x_top = _line_x(rho, sines[theta], cosines[theta], top)
            x_bottom = _line_x(rho, sines[theta], cosines[theta], bottom)
            anchor = x_top
            if (
                x_top < 0
                or x_top >= width
                or anchor < 0.12 * width
                or anchor > 0.88 * width
            ):
                continue
            candidate = _Candidate(
                theta=theta,
                rho=rho_index,
                votes=votes,
                x_top=x_top,
                x_bottom=x_bottom,
            )
            candidate.support = _edge_support(
                evidence, candidate, top, bottom
            )
            if candidate.support < 0.45:
                continue
            is_left = anchor < width * 0.5
            candidate.orientation = _orientation_score(
                x_top, x_bottom, top, bottom, is_left
            )
            vote_score = min(1.0, votes / (bottom - top + 1))
            candidate.score = (
                0.55 * candidate.orientation
                + 0.35 * candidate.support
                + 0.10 * vote_score
            )
            _add_candidate(left if is_left else right, candidate)

    if left:
        _draw_candidate(hough_lines, left[0], top, bottom)
    if right:
        _draw_candidate(hough_lines, right[0], top, bottom)

    left_confidence = left[0].score if left else 0.0
    right_confidence = right[0].score if right else 0.0
    selected_left = None
    selected_right = None
    best_score = -1.0
    for left_candidate in left:
        for right_candidate in right:
            geometry = _pair_geometry(
                left_candidate, right_candidate, width, top, bottom
            )
            if geometry < 0:
                continue
            score = (
                0.40 * left_candidate.score
                + 0.40 * right_candidate.score
                + 0.20 * geometry
            )
            if score > best_score:
                best_score = score
                selected_left = left_candidate
                selected_right = right_candidate

    if selected_left is not None and best_score >= 0.65:
        _draw_candidate(boundary_mask, selected_left, top, bottom)
        _draw_candidate(boundary_mask, selected_right, top, bottom)
        left_slope = (
            (selected_left.x_bottom - selected_left.x_top) / (bottom - top)
        )
        right_slope = (
            (selected_right.x_bottom - selected_right.x_top) / (bottom - top)
        )
        for y in range(top, bottom + 1):
            start = max(0, min(
                width - 1,
                math.ceil(selected_left.x_top + (y - top) * left_slope),
            ))
            end = max(0, min(
                width - 1,
                math.floor(selected_right.x_top + (y - top) * right_slope),
            ))
            row_is_road = road_mask[y, start:end + 1] != 0
            traversable_mask[y, start:end + 1][row_is_road] = 255
        result = PerceptionResult(
            validity=VALID_PAIR,
            confidence=float(np.float32(best_score)),
            left_confidence=float(np.float32(selected_left.score)),
            right_confidence=float(np.float32(selected_right.score)),
        )
    elif left and right:
        validity = LOW_CONFIDENCE if best_score >= 0 else PAIR_REJECTED
        result = PerceptionResult(
            validity=validity,
            confidence=float(np.float32(max(0.0, best_score))),
            left_confidence=float(np.float32(left_confidence)),
            right_confidence=float(np.float32(right_confidence)),
        )
    elif left or right:
        candidate = left[0] if left else right[0]
        _draw_candidate(boundary_mask, candidate, top, bottom)
        result = PerceptionResult(
            validity=LEFT_ONLY if left else RIGHT_ONLY,
            confidence=float(np.float32(candidate.score)),
            left_confidence=float(np.float32(left_confidence)),
            right_confidence=float(np.float32(right_confidence)),
        )
    else:
        result = PerceptionResult()

    return result, hough_lines, boundary_mask, traversable_mask


def process_traversability_pipeline(rgb):
    """Run the unchanged FCV algorithm on one RGB8 NumPy image."""
    _validate_rgb8(rgb)
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    vehicle_mask = _vehicle_mask(width, height)
    gray = rgb8_to_gray(rgb)
    denoised = cv2.blur(
        gray, (3, 3), borderType=cv2.BORDER_REPLICATE
    )
    edges = _sobel_edges(denoised)
    road_mask = _road_candidate_mask(rgb, gray, vehicle_mask)
    evidence = _color_boundary_evidence(road_mask, edges)
    result, hough_lines, boundary_mask, traversable_mask = (
        _detect_boundaries(evidence, road_mask)
    )
    boundary_mask[vehicle_mask] = 0
    traversable_mask[vehicle_mask] = 0

    outputs = PipelineOutputs(
        gray=gray,
        denoised=denoised,
        edges=edges,
        road_mask=road_mask,
        hough_lines=hough_lines,
        boundary_mask=boundary_mask,
        traversable_mask=traversable_mask,
    )
    return result, outputs
