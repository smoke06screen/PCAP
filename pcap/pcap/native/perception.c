#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>

#include <omp.h>

static inline int clamp_int(int value, int low, int high)
{
    if (value < low) {
        return low;
    }

    if (value > high) {
        return high;
    }

    return value;
}

static inline int max_int(int a, int b)
{
    return (a > b) ? a : b;
}

static inline int min_int(int a, int b)
{
    return (a < b) ? a : b;
}

static inline int is_vehicle_pixel(int x, int y, int width, int height)
{
    const int vehicle_top = (height * 78) / 100;

    if (y < vehicle_top) {
        return 0;
    }

    const int vehicle_bottom = height - 1;
    const float t = (vehicle_bottom > vehicle_top)
        ? (float)(y - vehicle_top) / (float)(vehicle_bottom - vehicle_top)
        : 1.0f;

    const float center = (float)width * 0.5f;
    const float top_half_width = (float)width * 0.12f;
    const float bottom_half_width = (float)width * 0.21f;
    const float half_width = top_half_width + (bottom_half_width - top_half_width) * t;

    return (x >= (int)(center - half_width) && x <= (int)(center + half_width));
}

static void clear_vehicle_region(
    uint8_t *boundary_mask,
    uint8_t *traversable_mask,
    int width,
    int height
)
{
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (is_vehicle_pixel(x, y, width, height)) {
                const size_t idx = (size_t)y * (size_t)width + (size_t)x;

                if (boundary_mask != NULL) {
                    boundary_mask[idx] = 0;
                }

                if (traversable_mask != NULL) {
                    traversable_mask[idx] = 0;
                }
            }
        }
    }
}

static void rgb8_to_gray(
    const uint8_t *input,
    uint8_t *gray,
    int width,
    int height
)
{
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t src_idx = ((size_t)y * (size_t)width + (size_t)x) * 3U;
            const size_t dst_idx = (size_t)y * (size_t)width + (size_t)x;

            const uint8_t r = input[src_idx];
            const uint8_t g = input[src_idx + 1U];
            const uint8_t b = input[src_idx + 2U];

            const float luminance =
                0.299f * (float)r +
                0.587f * (float)g +
                0.114f * (float)b;

            gray[dst_idx] = (uint8_t)(luminance + 0.5f);
        }
    }
}

static void box_blur_3x3(
    const uint8_t *input,
    uint8_t *output,
    int width,
    int height
)
{
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            int count = 0;

            for (int dy = -1; dy <= 1; ++dy) {
                const int yy = clamp_int(y + dy, 0, height - 1);

                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = clamp_int(x + dx, 0, width - 1);
                    const size_t idx = (size_t)yy * (size_t)width + (size_t)xx;

                    sum += (int)input[idx];
                    ++count;
                }
            }

            output[(size_t)y * (size_t)width + (size_t)x] =
                (uint8_t)((sum + (count / 2)) / count);
        }
    }
}

static void sobel_edges(
    const uint8_t *input,
    uint8_t *output,
    int width,
    int height
)
{
    const int threshold = 120;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t out_idx = (size_t)y * (size_t)width + (size_t)x;

            if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
                output[out_idx] = 0;
                continue;
            }

            const int p00 = input[(size_t)(y - 1) * (size_t)width + (size_t)(x - 1)];
            const int p01 = input[(size_t)(y - 1) * (size_t)width + (size_t)x];
            const int p02 = input[(size_t)(y - 1) * (size_t)width + (size_t)(x + 1)];
            const int p10 = input[(size_t)y * (size_t)width + (size_t)(x - 1)];
            const int p12 = input[(size_t)y * (size_t)width + (size_t)(x + 1)];
            const int p20 = input[(size_t)(y + 1) * (size_t)width + (size_t)(x - 1)];
            const int p21 = input[(size_t)(y + 1) * (size_t)width + (size_t)x];
            const int p22 = input[(size_t)(y + 1) * (size_t)width + (size_t)(x + 1)];

            const int gx =
                -p00 + p02 -
                (2 * p10) + (2 * p12) -
                p20 + p22;

            const int gy =
                p00 + (2 * p01) + p02 -
                p20 - (2 * p21) - p22;

            const int magnitude = abs(gx) + abs(gy);

            output[out_idx] = (magnitude > threshold) ? 255 : 0;
        }
    }
}

static void build_road_candidate_mask(
    const uint8_t *input,
    const uint8_t *gray,
    uint8_t *road_mask,
    int width,
    int height
)
{
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = (size_t)y * (size_t)width + (size_t)x;

            const size_t rgb_idx = idx * 3U;
            const int r = input[rgb_idx];
            const int g = input[rgb_idx + 1U];
            const int b = input[rgb_idx + 2U];
            const int brightness = gray[idx];
            const int max_rgb = max_int(r, max_int(g, b));
            const int min_rgb = min_int(r, min_int(g, b));
            const int saturation = max_rgb - min_rgb;
            const int green_excess = g - ((r + b) / 2);

            if (is_vehicle_pixel(x, y, width, height)) {
                road_mask[idx] = 0;
                continue;
            }

            road_mask[idx] =
                (brightness > 35 && brightness < 215 &&
                 saturation < 90 && green_excess < 18) ? 255 : 0;
        }
    }
}

/* The center-connected color region supplies boundary points without
 * letting asphalt texture or disconnected grass patches dominate Hough.
 */
static void build_color_boundary_evidence(
    const uint8_t *road_mask,
    uint8_t *evidence,
    uint8_t *debug_edges,
    int width,
    int height
)
{
    const int top = (height * 25) / 100;
    const int bottom = (height * 78) / 100 - 1;
    const int center = width / 2;
    for (int y = top; y <= bottom; ++y) {
        const size_t row = (size_t)y * (size_t)width;
        if (road_mask[row + (size_t)center] == 0) continue;
        int left = center, right = center;
        while (left > 0 && road_mask[row + (size_t)(left - 1)] != 0) --left;
        while (right < width - 1 && road_mask[row + (size_t)(right + 1)] != 0) ++right;
        if (right - left < width / 12) continue;
        if (left > 0) {
            evidence[row + (size_t)left] = 255;
            debug_edges[row + (size_t)left] = 255;
        }
        if (right < width - 1) {
            evidence[row + (size_t)right] = 255;
            debug_edges[row + (size_t)right] = 255;
        }
    }
}

/* Scores are evidence weights, not calibrated probabilities. */
enum { NO_BOUNDARY = 0, LEFT_ONLY = 1, RIGHT_ONLY = 2,
       VALID_PAIR = 3, PAIR_REJECTED = 4, LOW_CONFIDENCE = 5 };
enum { THETA_MIN = 0, THETA_BINS = 180,
       MAX_CANDIDATES = 24, SUPPORT_RADIUS = 2 };

typedef struct {
    int validity;
    float confidence;
    float left_confidence;
    float right_confidence;
} PerceptionResult;

typedef struct {
    int theta;
    int rho;
    int votes;
    double x_top;
    double x_bottom;
    double orientation;
    double support;
    double score;
} Candidate;

static double line_x(double rho, double sine, double cosine, int y)
{
    return (rho - (double)y * sine) / cosine;
}

/* Prefer a modest outward slant toward the bottom of the image.
 * A vertical line remains plausible; a steep or wrong-way line scores less.
 */
static double orientation_score(double x_top, double x_bottom,
                                int top, int bottom, int is_left)
{
    const double slope = (x_bottom - x_top) / (double)(bottom - top);
    const double magnitude = fabs(slope);
    double score;
    if (magnitude <= 0.35) {
        score = 0.55 + 0.45 * magnitude / 0.35;
    } else {
        score = 1.0 - 0.40 * fmin(1.0, (magnitude - 0.35) / 1.65);
    }
    if ((is_left && slope > 0.05) || (!is_left && slope < -0.05)) {
        score *= 0.55;
    }
    return score;
}

static void draw_candidate(uint8_t *mask, const Candidate *candidate,
                           int width, int top, int bottom)
{
    const double slope = (candidate->x_bottom - candidate->x_top) /
                         (double)(bottom - top);
    for (int y = top; y <= bottom; ++y) {
        const int x = (int)lround(candidate->x_top + (y - top) * slope);
        if (x >= 0 && x < width) {
            mask[(size_t)y * (size_t)width + (size_t)x] = 255;
        }
    }
}

static double edge_support(const uint8_t *edges, const Candidate *candidate,
                           int width, int top, int bottom)
{
    int supported = 0;
    int visible = 0;
    const int samples = bottom - top + 1;
    const double slope = (candidate->x_bottom - candidate->x_top) /
                         (double)(bottom - top);
    for (int y = top; y <= bottom; ++y) {
        const int x = (int)lround(candidate->x_top + (y - top) * slope);
        if (x < 0 || x >= width) continue;
        ++visible;
        for (int dx = -SUPPORT_RADIUS; dx <= SUPPORT_RADIUS; ++dx) {
            const int xx = x + dx;
            if (xx >= 0 && xx < width &&
                edges[(size_t)y * (size_t)width + (size_t)xx] != 0) {
                ++supported;
                break;
            }
        }
    }
    if (visible < samples / 5) return 0.0;
    return (double)supported / (double)visible;
}

static void add_candidate(Candidate *list, int *count, Candidate candidate)
{
    int place = *count;
    if (place == MAX_CANDIDATES) {
        if (candidate.score <= list[place - 1].score) {
            return;
        }
        --place;
    } else {
        ++*count;
    }
    while (place > 0 && list[place - 1].score < candidate.score) {
        list[place] = list[place - 1];
        --place;
    }
    list[place] = candidate;
}

static int is_local_peak(const int *accumulator, int t, int r, int rho_bins)
{
    const int votes = accumulator[(size_t)t * (size_t)rho_bins + (size_t)r];
    for (int dt = -2; dt <= 2; ++dt) {
        const int tt = t + dt;
        if (tt < 0 || tt >= THETA_BINS) continue;
        for (int dr = -3; dr <= 3; ++dr) {
            const int rr = r + dr;
            if (rr < 0 || rr >= rho_bins || (dt == 0 && dr == 0)) continue;
            const int neighbor = accumulator[(size_t)tt * (size_t)rho_bins +
                                             (size_t)rr];
            if (neighbor > votes ||
                (neighbor == votes && (tt < t || (tt == t && rr < r)))) {
                return 0;
            }
        }
    }
    return 1;
}

static double pair_geometry(const Candidate *left, const Candidate *right,
                            int width, int top, int bottom)
{
    const double left_slope = (left->x_bottom - left->x_top) / (bottom - top);
    const double right_slope = (right->x_bottom - right->x_top) / (bottom - top);
    const double top_width = right->x_top - left->x_top;
    int visible_rows = 0;
    double last_left = 0.0, last_right = 0.0;
    for (int y = top; y <= bottom; ++y) {
        const double lx = left->x_top + (y - top) * left_slope;
        const double rx = right->x_top + (y - top) * right_slope;
        if (lx >= 0.0 && lx < width && rx >= 0.0 && rx < width) {
            ++visible_rows;
            last_left = lx;
            last_right = rx;
        }
    }
    const double bottom_width = last_right - last_left;
    if (visible_rows < (bottom - top + 1) / 5 ||
        top_width < 0.08 * width || bottom_width < 0.18 * width ||
        top_width > 0.90 * width || bottom_width > 0.98 * width ||
        bottom_width < 0.75 * top_width) {
        return -1.0;
    }
    const double center = (last_left + last_right) * 0.5;
    const double center_score = 1.0 - fmin(1.0, fabs(center / width - 0.5) / 0.5);
    const double convergence = fmin(1.0, bottom_width / top_width);
    return 0.5 * center_score + 0.5 * convergence;
}

static int detect_boundaries_and_traversable(
    const uint8_t *edges,
    const uint8_t *road_mask,
    uint8_t *hough_lines,
    uint8_t *boundary_mask,
    uint8_t *traversable_mask,
    int width,
    int height,
    PerceptionResult *result
)
{
    const int top = (height * 25) / 100;
    const int bottom = (height * 78) / 100 - 1;
    const int diag = (int)ceil(hypot((double)width, (double)height));
    const int rho_bins = (diag * 2) + 1;
    const int rho_offset = diag;
    const double radians_per_degree = 3.14159265358979323846 / 180.0;
    double cos_table[THETA_BINS];
    double sin_table[THETA_BINS];
    Candidate left[MAX_CANDIDATES], right[MAX_CANDIDATES];
    int left_count = 0, right_count = 0;
    const size_t pixels = (size_t)width * (size_t)height;
    memset(boundary_mask, 0, pixels);
    memset(traversable_mask, 0, pixels);
    if (hough_lines != NULL) memset(hough_lines, 0, pixels);
    if (bottom <= top || width < 8) return 0;

    int *accumulator = (int *)calloc((size_t)THETA_BINS * (size_t)rho_bins,
                                     sizeof(int));

    if (accumulator == NULL) {
        return -2;
    }

    for (int t = 0; t < THETA_BINS; ++t) {
        const double theta = (double)(THETA_MIN + t) * radians_per_degree;
        cos_table[t] = cos(theta);
        sin_table[t] = sin(theta);
    }

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = (size_t)y * (size_t)width + (size_t)x;
            if (edges[idx] == 0 || road_mask[idx] == 0) {
                continue;
            }

            if (y < top || y > bottom) continue;
            for (int t = 0; t < THETA_BINS; ++t) {
                const double rho = ((double)x * cos_table[t]) + ((double)y * sin_table[t]);
                const int rho_idx = (int)(rho + (double)rho_offset + 0.5);

                if (rho_idx >= 0 && rho_idx < rho_bins) {
                    #pragma omp atomic
                    accumulator[(size_t)t * (size_t)rho_bins + (size_t)rho_idx] += 1;
                }
            }
        }
    }

    for (int t = 0; t < THETA_BINS; ++t) {
        const double verticality = fabs(cos_table[t]);
        /* Horizontal lines have near-zero verticality and receive no candidacy. */
        if (verticality < 0.35) continue;
        for (int r = 0; r < rho_bins; ++r) {
            const int votes = accumulator[(size_t)t * (size_t)rho_bins + (size_t)r];
            if (votes < (bottom - top + 1) * 12 / 100 ||
                !is_local_peak(accumulator, t, r, rho_bins)) continue;
            const double rho = (double)r - (double)rho_offset;
            Candidate candidate = {0};
            candidate.theta = t;
            candidate.rho = r;
            candidate.votes = votes;
            candidate.x_top = line_x(rho, sin_table[t], cos_table[t], top);
            candidate.x_bottom = line_x(rho, sin_table[t], cos_table[t], bottom);
            const double anchor = candidate.x_top;
            if (candidate.x_top < 0 || candidate.x_top >= width ||
                anchor < 0.12 * width || anchor > 0.88 * width) continue;
            candidate.support = edge_support(edges, &candidate, width, top, bottom);
            if (candidate.support < 0.45) continue;
            const int is_left = anchor < width * 0.5;
            candidate.orientation = orientation_score(
                candidate.x_top, candidate.x_bottom, top, bottom, is_left);
            const double vote_score = fmin(1.0, (double)votes / (bottom - top + 1));
            candidate.score = 0.55 * candidate.orientation +
                              0.35 * candidate.support + 0.10 * vote_score;
            if (is_left) {
                add_candidate(left, &left_count, candidate);
            } else {
                add_candidate(right, &right_count, candidate);
            }
        }
    }
    if (hough_lines != NULL) {
        if (left_count) draw_candidate(hough_lines, &left[0], width, top, bottom);
        if (right_count) draw_candidate(hough_lines, &right[0], width, top, bottom);
    }
    result->left_confidence = left_count ? (float)left[0].score : 0.0f;
    result->right_confidence = right_count ? (float)right[0].score : 0.0f;
    int selected_left = -1, selected_right = -1;
    double best_score = -1.0;
    for (int i = 0; i < left_count; ++i) {
        for (int j = 0; j < right_count; ++j) {
            const double geometry = pair_geometry(&left[i], &right[j],
                                                   width, top, bottom);
            if (geometry < 0) continue;
            const double score = 0.40 * left[i].score +
                                 0.40 * right[j].score + 0.20 * geometry;
            if (score > best_score) {
                best_score = score;
                selected_left = i;
                selected_right = j;
            }
        }
    }
    if (selected_left >= 0 && best_score >= 0.65) {
        const Candidate *l = &left[selected_left], *r = &right[selected_right];
        result->validity = VALID_PAIR;
        result->confidence = (float)best_score;
        result->left_confidence = (float)l->score;
        result->right_confidence = (float)r->score;
        draw_candidate(boundary_mask, l, width, top, bottom);
        draw_candidate(boundary_mask, r, width, top, bottom);
        const double left_slope = (l->x_bottom - l->x_top) / (bottom - top);
        const double right_slope = (r->x_bottom - r->x_top) / (bottom - top);
        #pragma omp parallel for schedule(static)
        for (int y = top; y <= bottom; ++y) {
            const int start = clamp_int((int)ceil(l->x_top + (y - top) * left_slope), 0, width - 1);
            const int end = clamp_int((int)floor(r->x_top + (y - top) * right_slope), 0, width - 1);
            for (int x = start; x <= end; ++x) {
                const size_t idx = (size_t)y * (size_t)width + (size_t)x;
                if (road_mask[idx]) traversable_mask[idx] = 255;
            }
        }
    } else if (left_count && right_count) {
        result->validity = (best_score >= 0) ? LOW_CONFIDENCE : PAIR_REJECTED;
        result->confidence = (float)fmax(0.0, best_score);
    } else if (left_count || right_count) {
        const Candidate *single = left_count ? &left[0] : &right[0];
        result->validity = left_count ? LEFT_ONLY : RIGHT_ONLY;
        result->confidence = (float)single->score;
        draw_candidate(boundary_mask, single, width, top, bottom);
    }
    free(accumulator);
    return 0;
}

int process_traversability_pipeline_ex(
    const uint8_t *input,
    uint8_t *gray,
    uint8_t *denoised,
    uint8_t *edges,
    uint8_t *road_mask,
    uint8_t *hough_lines,
    uint8_t *boundary_mask,
    uint8_t *traversable_mask,
    int width,
    int height,
    PerceptionResult *result
)
{
    if (input == NULL || gray == NULL || result == NULL || width <= 0 || height <= 0 ||
        width > INT_MAX / 100 || height > INT_MAX / 100 ||
        (size_t)width > SIZE_MAX / (size_t)height / 3U ||
        hypot((double)width, (double)height) > (INT_MAX - 1) / 2.0) {
        return -1;
    }
    *result = (PerceptionResult){0};

    rgb8_to_gray(input, gray, width, height);

    if (denoised != NULL) {
        box_blur_3x3(gray, denoised, width, height);
    }

    if (edges != NULL || road_mask != NULL || hough_lines != NULL || boundary_mask != NULL || traversable_mask != NULL) {
        if (edges == NULL || road_mask == NULL || boundary_mask == NULL ||
            traversable_mask == NULL) {
            return -1;
        }

        const uint8_t *edge_source = (denoised != NULL) ? denoised : gray;
        sobel_edges(edge_source, edges, width, height);

        build_road_candidate_mask(input, gray, road_mask, width, height);
        const size_t pixels = (size_t)width * (size_t)height;
        uint8_t *boundary_evidence = (uint8_t *)calloc(pixels, 1U);
        if (boundary_evidence == NULL) return -2;
        build_color_boundary_evidence(road_mask, boundary_evidence, edges,
                                      width, height);

        const int status = detect_boundaries_and_traversable(
            boundary_evidence,
            road_mask,
            hough_lines,
            boundary_mask,
            traversable_mask,
            width,
            height,
            result
        );
        free(boundary_evidence);
        if (status != 0) return status;
        clear_vehicle_region(boundary_mask, traversable_mask, width, height);
    }

    return 0;
}

/* Preserve the original ctypes ABI for existing callers. */
int process_traversability_pipeline(
    const uint8_t *input, uint8_t *gray, uint8_t *denoised,
    uint8_t *edges, uint8_t *road_mask, uint8_t *hough_lines,
    uint8_t *boundary_mask, uint8_t *traversable_mask,
    int width, int height)
{
    PerceptionResult ignored;
    return process_traversability_pipeline_ex(
        input, gray, denoised, edges, road_mask, hough_lines,
        boundary_mask, traversable_mask, width, height, &ignored);
}

int process_rgb8_to_gray(
    const uint8_t *input,
    uint8_t *output,
    int width,
    int height
)
{
    return process_traversability_pipeline(
        input,
        output,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        width,
        height
    );
}
