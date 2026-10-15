#include <stdint.h>
#include <stddef.h>
#include <omp.h>

/*
 * Convert an RGB8 image to an 8-bit grayscale image.
 *
 * input:
 *   RGBRGBRGB...
 *
 * output:
 *   one uint8 grayscale value per pixel
 *
 * Returns:
 *   0  -> success
 *  -1  -> invalid arguments
 */
int process_rgb8_to_gray(
    const uint8_t *input,
    uint8_t *output,
    int width,
    int height
)
{
    if (input == NULL || output == NULL || width <= 0 || height <= 0) {
        return -1;
    }

    const size_t pixels = (size_t)width * (size_t)height;

    /*
     * Each pixel is independent, so this is naturally
     * parallelizable with OpenMP.
     */
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < pixels; ++i) {

        const size_t idx = i * 3;

        const uint8_t r = input[idx];
        const uint8_t g = input[idx + 1];
        const uint8_t b = input[idx + 2];

        /*
         * Standard luminance approximation.
         */
        const float gray =
            0.299f * (float)r +
            0.587f * (float)g +
            0.114f * (float)b;

        output[i] = (uint8_t)(gray + 0.5f);
    }

    return 0;
}
