#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

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

static inline double abs_double(double value)
{
    return (value < 0.0) ? -value : value;
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

            const int road_score =
                ((brightness > 35 && brightness < 215) ? 1 : 0) +
                ((saturation < 90) ? 1 : 0) +
                ((green_excess < 18) ? 1 : 0);

            road_mask[idx] = (road_score >= 2) ? 255 : 0;
        }
    }
}

static void draw_line_segment(
    uint8_t *mask,
    int width,
    int height,
    int x0,
    int y0,
    int x1,
    int y1,
    uint8_t value
)
{
    int dx = abs(x1 - x0);
    int sx = (x0 < x1) ? 1 : -1;
    int dy = -abs(y1 - y0);
    int sy = (y0 < y1) ? 1 : -1;
    int error = dx + dy;

    while (1) {
        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            mask[(size_t)y0 * (size_t)width + (size_t)x0] = value;
        }

        if (x0 == x1 && y0 == y1) {
            break;
        }

        const int twice_error = 2 * error;

        if (twice_error >= dy) {
            error += dy;
            x0 += sx;
        }

        if (twice_error <= dx) {
            error += dx;
            y0 += sy;
        }
    }
}

static void detect_boundaries_and_traversable(
    const uint8_t *edges,
    const uint8_t *road_mask,
    uint8_t *hough_lines,
    uint8_t *boundary_mask,
    uint8_t *traversable_mask,
    int width,
    int height
)
{
    const int theta_min = 20;
    const int theta_max = 160;
    const int theta_step = 1;
    const int theta_bins = ((theta_max - theta_min) / theta_step) + 1;
    const int diag = (int)(sqrt((double)(width * width + height * height)) + 0.5);
    const int rho_bins = (diag * 2) + 1;
    const int rho_offset = diag;
    const double radians_per_degree = 3.14159265358979323846 / 180.0;

    double cos_table[theta_bins];
    double sin_table[theta_bins];
    int *accumulator = (int *)calloc((size_t)theta_bins * (size_t)rho_bins, sizeof(int));

    if (accumulator == NULL) {
        return;
    }

    for (int t = 0; t < theta_bins; ++t) {
        const double theta = (double)(theta_min + (t * theta_step)) * radians_per_degree;
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

            for (int t = 0; t < theta_bins; ++t) {
                const double rho = ((double)x * cos_table[t]) + ((double)y * sin_table[t]);
                const int rho_idx = (int)(rho + (double)rho_offset + 0.5);

                if (rho_idx >= 0 && rho_idx < rho_bins) {
                    #pragma omp atomic
                    accumulator[(size_t)t * (size_t)rho_bins + (size_t)rho_idx] += 1;
                }
            }
        }
    }

    int best_left_theta = -1;
    int best_left_rho = 0;
    int best_left_votes = -1;
    int best_right_theta = -1;
    int best_right_rho = 0;
    int best_right_votes = -1;

    for (int t = 0; t < theta_bins; ++t) {
        const double theta = (double)(theta_min + (t * theta_step)) * radians_per_degree;
        const double line_dx_dy = (abs_double(cos_table[t]) > 1e-6) ? (-sin_table[t] / cos_table[t]) : 0.0;

        for (int r = 0; r < rho_bins; ++r) {
            const int votes = accumulator[(size_t)t * (size_t)rho_bins + (size_t)r];

            if (votes <= 0) {
                continue;
            }

            const double rho = (double)r - (double)rho_offset;
            const double x_at_bottom = (abs_double(cos_table[t]) > 1e-6)
                ? ((rho - ((double)(height - 1) * sin_table[t])) / cos_table[t])
                : 0.0;
            const double x_at_top = (abs_double(cos_table[t]) > 1e-6)
                ? ((rho - 0.0 * sin_table[t]) / cos_table[t])
                : 0.0;

            const int bottom_x = (int)(x_at_bottom + 0.5);
            const int top_x = (int)(x_at_top + 0.5);
            const int candidate_is_left = (bottom_x < top_x);
            const int candidate_is_right = (bottom_x > top_x);

            if (candidate_is_left && votes > best_left_votes) {
                best_left_votes = votes;
                best_left_theta = t;
                best_left_rho = r;
            }

            if (candidate_is_right && votes > best_right_votes) {
                best_right_votes = votes;
                best_right_theta = t;
                best_right_rho = r;
            }

            (void)theta;
            (void)line_dx_dy;
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = (size_t)y * (size_t)width + (size_t)x;
            boundary_mask[idx] = 0;
            traversable_mask[idx] = 0;
            if (hough_lines != NULL) {
                hough_lines[idx] = 0;
            }
        }
    }

    if (best_left_theta >= 0) {
        const double theta = (double)(theta_min + (best_left_theta * theta_step)) * radians_per_degree;
        const double rho = (double)best_left_rho - (double)rho_offset;

        const int x0 = 0;
        const int y0 = clamp_int((int)(((rho - ((double)x0 * cos_table[best_left_theta])) / sin_table[best_left_theta]) + 0.5), 0, height - 1);
        const int x1 = width - 1;
        const int y1 = clamp_int((int)(((rho - ((double)x1 * cos_table[best_left_theta])) / sin_table[best_left_theta]) + 0.5), 0, height - 1);

        if (hough_lines != NULL) {
            draw_line_segment(hough_lines, width, height, x0, y0, x1, y1, 255);
        }

        draw_line_segment(boundary_mask, width, height, x0, y0, x1, y1, 255);

        for (int y = 0; y < height; ++y) {
            const int x = (abs_double(sin_table[best_left_theta]) > 1e-6)
                ? (int)(((rho - ((double)y * sin_table[best_left_theta])) / cos_table[best_left_theta]) + 0.5)
                : x0;

            if (x >= 0 && x < width) {
                const size_t idx = (size_t)y * (size_t)width + (size_t)x;
                if (road_mask[idx] != 0) {
                    traversable_mask[idx] = 255;
                }
            }
        }

        (void)theta;
    }

    if (best_right_theta >= 0) {
        const double theta = (double)(theta_min + (best_right_theta * theta_step)) * radians_per_degree;
        const double rho = (double)best_right_rho - (double)rho_offset;

        const int x0 = 0;
        const int y0 = clamp_int((int)(((rho - ((double)x0 * cos_table[best_right_theta])) / sin_table[best_right_theta]) + 0.5), 0, height - 1);
        const int x1 = width - 1;
        const int y1 = clamp_int((int)(((rho - ((double)x1 * cos_table[best_right_theta])) / sin_table[best_right_theta]) + 0.5), 0, height - 1);

        if (hough_lines != NULL) {
            draw_line_segment(hough_lines, width, height, x0, y0, x1, y1, 255);
        }

        draw_line_segment(boundary_mask, width, height, x0, y0, x1, y1, 255);

        for (int y = 0; y < height; ++y) {
            const int x = (abs_double(sin_table[best_right_theta]) > 1e-6)
                ? (int)(((rho - ((double)y * sin_table[best_right_theta])) / cos_table[best_right_theta]) + 0.5)
                : x1;

            if (x >= 0 && x < width) {
                const size_t idx = (size_t)y * (size_t)width + (size_t)x;
                if (road_mask[idx] != 0) {
                    traversable_mask[idx] = 255;
                }
            }
        }

        (void)theta;
    }

    free(accumulator);
}

int process_traversability_pipeline(
    const uint8_t *input,
    uint8_t *gray,
    uint8_t *denoised,
    uint8_t *edges,
    uint8_t *road_mask,
    uint8_t *hough_lines,
    uint8_t *boundary_mask,
    uint8_t *traversable_mask,
    int width,
    int height
)
{
    if (input == NULL || gray == NULL || width <= 0 || height <= 0) {
        return -1;
    }

    rgb8_to_gray(input, gray, width, height);

    if (denoised != NULL) {
        box_blur_3x3(gray, denoised, width, height);
    }

    if (edges != NULL || road_mask != NULL || hough_lines != NULL || boundary_mask != NULL || traversable_mask != NULL) {
        if (road_mask == NULL || boundary_mask == NULL || traversable_mask == NULL) {
            return -1;
        }

        if (edges != NULL) {
            const uint8_t *edge_source = (denoised != NULL) ? denoised : gray;
            sobel_edges(edge_source, edges, width, height);
        }

        build_road_candidate_mask(input, gray, road_mask, width, height);

        detect_boundaries_and_traversable(
            (edges != NULL) ? edges : gray,
            road_mask,
            hough_lines,
            boundary_mask,
            traversable_mask,
            width,
            height
        );

        clear_vehicle_region(boundary_mask, traversable_mask, width, height);
    }

    return 0;
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
