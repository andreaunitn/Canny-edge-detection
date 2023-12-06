/* COMP.CE.350 Parallelization Excercise 2023
   Copyright (c) 2023 Topi Leppanen topi.leppanen@tuni.fi
                      Jan Solanti

VERSION 23.0 - Created
*/

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "util.h"

// Is used to find out frame times
int previousFinishTime = 0;
unsigned int frameNumber = 0;
unsigned int seed = 0;

typedef struct {
    uint16_t x;
    uint16_t y;
} coord_t;

const coord_t neighbour_offsets[8] = {
    {-1, -1}, {0, -1},  {+1, -1}, {-1, 0},
    {+1, 0},  {-1, +1}, {0, +1},  {+1, +1},
};

// ## You may add your own variables here ##

// Utility function to convert 2d index with offset to linear index
// Uses clamp-to-edge out-of-bounds handling
size_t
idx(size_t x, size_t y, size_t width, size_t height, int xoff, int yoff) {
    size_t resx = x;

    bool is_positive_x_off = xoff > 0;
    bool is_valid_positive_range_x = is_positive_x_off && x < (width - xoff);
    bool is_valid_negative_range_x = !is_positive_x_off && x >= -xoff;

    bool final_condition_x = (is_valid_positive_range_x || is_valid_negative_range_x);
    resx += final_condition_x ? xoff : 0;

    size_t resy = y;

    bool is_positive_y_off = yoff > 0;
    bool is_valid_positive_range_y = is_positive_y_off && y < (height - yoff);
    bool is_valid_negative_range_y = !is_positive_y_off && y >= -yoff;

    bool final_condition_y = is_valid_positive_range_y || is_valid_negative_range_y;
    resy += final_condition_y ? yoff : 0;

    return resy * width + resx;

}

void sobel3x3(const uint8_t *restrict in, size_t width, size_t height,
              int16_t *restrict output_x, int16_t *restrict output_y) {

    size_t tileWidth = 16;
    size_t tileHeight = 16;

    // LOOP 1.0.1
    #pragma omp parallel for
    for (size_t tileY = 0; tileY < height; tileY += tileHeight) {
        // LOOP 1.0.2
        #pragma omp parallel for
        for (size_t tileX = 0; tileX < width; tileX += tileWidth) {

            size_t actualTileWidth = (tileX + tileWidth > width) ? (width - tileX) : tileWidth;
            size_t actualTileHeight = (tileY + tileHeight > height) ? (height - tileY) : tileHeight;

            // LOOP 1.1
            for (size_t y = tileY; y < tileY + actualTileHeight; y++) {
                // LOOP 1.2
                for (size_t x = tileX; x < tileX + actualTileWidth; x++) {
                    size_t gid = y * width + x;

                    // 3x3 sobel filter, first in x direction
                    output_x[gid] = (-1) * in[idx(x, y, width, height, -1, -1)] +
                                 1 * in[idx(x, y, width, height, 1, -1)] +
                                (-2) * in[idx(x, y, width, height, -1, 0)] +
                                 2 * in[idx(x, y, width, height, 1, 0)] +
                                  (-1) * in[idx(x, y, width, height, -1, 1)] +
                                   1 * in[idx(x, y, width, height, 1, 1)];

                    // 3x3 sobel filter, in y direction
                    output_y[gid] = (-1) * in[idx(x, y, width, height, -1, -1)] +
                                  1 * in[idx(x, y, width, height, -1, 1)] +
                                  (-2) * in[idx(x, y, width, height, 0, -1)] +
                                  2 * in[idx(x, y, width, height, 0, 1)] +
                                  (-1) * in[idx(x, y, width, height, 1, -1)] +
                                  1 * in[idx(x, y, width, height, 1, 1)];
                }
            }
        }
    }
}

float fast_atan2f(float y, float x) {
    float a, r, s, t, c, q, ax, ay, mx, mn;
    ax = fabsf (x);
    ay = fabsf (y);
    mx = fmaxf (ay, ax);
    mn = fminf (ay, ax);
    a = mn / mx;

    s = a * a;
    c = s * a;
    q = s * s;
    r = 0.024840285f * q + 0.18681418f;
    t = -0.094097948f * q - 0.33213072f;
    r = r * s + t;
    r = r * c + a;

    /* Map to full circle */
    bool cond1 = ay > ax;
    r = cond1 ? (PI/2.0 - r) : r;

    bool cond2 = x < 0;
    r = cond2 ? (PI - r) : r;

    bool cond3 = y < 0;
    r = cond3 ? -r : r;

    return r;
}

void
phaseAndMagnitude(
    const int16_t *restrict in_x, const int16_t *restrict in_y, size_t width,
    size_t height, uint8_t *restrict phase_out,
    uint16_t *restrict magnitude_out) {

    size_t tileWidth = 4;
    size_t tileHeight = 4;

    // LOOP 2.0.1
    #pragma omp parallel for
    for (size_t tileY = 0; tileY < height; tileY += tileHeight) {
        // LOOP 2.0.2
        #pragma omp parallel for
        for (size_t tileX = 0; tileX < width; tileX += tileWidth) {

            size_t actualTileWidth = (tileX + tileWidth > width) ? (width - tileX) : tileWidth;
            size_t actualTileHeight = (tileY + tileHeight > height) ? (height - tileY) : tileHeight;

            // LOOP 2.1.1
            for (size_t y = tileY; y < tileY + actualTileHeight; y++) {
                // LOOP 2.1.2
                for (size_t x = tileX; x < tileX + actualTileWidth; x+=4) {
                    size_t gid = y * width + x;
                    size_t gid2 = y * width + x + 1;
                    size_t gid3 = y * width + x + 2;
                    size_t gid4 = y * width + x + 3;

                    // Output in range -PI:PI
                    float angle = fast_atan2f(in_y[gid], in_x[gid]);
                    float angle2 = fast_atan2f(in_y[gid2], in_x[gid2]);
                    float angle3 = fast_atan2f(in_y[gid3], in_x[gid3]);
                    float angle4 = fast_atan2f(in_y[gid4], in_x[gid4]);

                    // Shift range -1:1
                    angle = (angle / PI) * 127.5 + 127.5 + 0.5;
                    angle2 = (angle2 / PI) * 127.5 + 127.5 + 0.5;
                    angle3 = (angle3 / PI) * 127.5 + 127.5 + 0.5;
                    angle4 = (angle4 / PI) * 127.5 + 127.5 + 0.5;

                    // Downcasting truncates angle to range 0:255
                    phase_out[gid] = (uint8_t)angle;
                    phase_out[gid2] = (uint8_t)angle2;
                    phase_out[gid3] = (uint8_t)angle3;
                    phase_out[gid4] = (uint8_t)angle4;
                }
            }

            // LOOP 2.2.1
            for (size_t y = tileY; y < tileY + actualTileHeight; y++) {
                // LOOP 2.2.2
                for (size_t x = tileX; x < tileX + actualTileWidth; x+=4) {
                    size_t gid = y * width + x;
                    size_t gid2 = y * width + x + 1;
                    size_t gid3 = y * width + x + 2;
                    size_t gid4 = y * width + x + 3;

                    magnitude_out[gid] = abs(in_x[gid]) + abs(in_y[gid]);
                    magnitude_out[gid2] = abs(in_x[gid2]) + abs(in_y[gid2]);
                    magnitude_out[gid3] = abs(in_x[gid3]) + abs(in_y[gid3]);
                    magnitude_out[gid4] = abs(in_x[gid4]) + abs(in_y[gid4]);
                }
            }
        }
    }
}

void
nonMaxSuppression(
    const uint16_t *restrict magnitude, const uint8_t *restrict phase,
    size_t width, size_t height, int16_t threshold_lower,
    uint16_t threshold_upper, uint8_t *restrict out) {

    #pragma omp parallel for
    // LOOP 3.1
    for (size_t y = 0; y < height; y++) {
        // LOOP 3.2
        for (size_t x = 0; x < width; x++) {
            size_t gid = y * width + x;

            uint8_t sobel_angle = phase[gid];

            if (sobel_angle > 127) {
                sobel_angle -= 128;
            }

            int sobel_orientation = 0;

            if (sobel_angle < 16 || sobel_angle >= (7 * 16)) {
                sobel_orientation = 2;
            } else if (sobel_angle >= 16 && sobel_angle < 16 * 3) {
                sobel_orientation = 1;
            } else if (sobel_angle >= 16 * 3 && sobel_angle < 16 * 5) {
                sobel_orientation = 0;
            } else if (sobel_angle > 16 * 5 && sobel_angle <= 16 * 7) {
                sobel_orientation = 3;
            }

            uint16_t sobel_magnitude = magnitude[gid];
            /* Non-maximum suppression
             * Pick out the two neighbours that are perpendicular to the
             * current edge pixel */
            uint16_t neighbour_max = 0;
            uint16_t neighbour_max2 = 0;
            switch (sobel_orientation) {
                case 0:
                    neighbour_max =
                        magnitude[idx(x, y, width, height, 0, -1)];
                    neighbour_max2 =
                        magnitude[idx(x, y, width, height, 0, 1)];
                    break;
                case 1:
                    neighbour_max =
                        magnitude[idx(x, y, width, height, -1, -1)];
                    neighbour_max2 =
                        magnitude[idx(x, y, width, height, 1, 1)];
                    break;
                case 2:
                    neighbour_max =
                        magnitude[idx(x, y, width, height, -1, 0)];
                    neighbour_max2 =
                        magnitude[idx(x, y, width, height, 1, 0)];
                    break;
                case 3:
                default:
                    neighbour_max =
                        magnitude[idx(x, y, width, height, 1, -1)];
                    neighbour_max2 =
                        magnitude[idx(x, y, width, height, -1, 1)];
                    break;
            }
            // Suppress the pixel here
            if ((sobel_magnitude < neighbour_max) ||
                (sobel_magnitude < neighbour_max2)) {
                sobel_magnitude = 0;
            }

            /* Double thresholding */
            // Marks YES pixels with 255, NO pixels with 0 and MAYBE pixels
            // with 127
            uint8_t t = 127;
            if (sobel_magnitude > threshold_upper) t = 255;
            if (sobel_magnitude <= threshold_lower) t = 0;
            out[gid] = t;
        }
    }
}

void
edgeTracing(uint8_t *restrict image, size_t width, size_t height) {

    coord_t *yes_pixels = malloc(width * height * sizeof(coord_t));
    size_t num_yes_pixels = 0;

    #pragma omp parallel for
    // LOOP 4.1
    for(uint16_t y = 0; y < height; y++) {
        // LOOP 4.2
        for(uint16_t x = 0; x < width; x++) {

            if(image[y * width + x] == 255) {

                size_t index = num_yes_pixels;

                #pragma omp critical
                {
                    index = num_yes_pixels++;
                }

                yes_pixels[index] = (coord_t){x, y};
            }
        }
    }

    bool new_yes_found;

    // LOOP 4.3.1
    #pragma omp parallel
    do {
            #pragma omp critical
            {
                new_yes_found = false;

                coord_t *new_yes_pixels = malloc(width * height * sizeof(coord_t));
                size_t num_new_yes = 0;

                // LOOP 4.3.2
                for(size_t i = 0; i < num_yes_pixels; i++) {
                    coord_t yes_pixel = yes_pixels[i];

                    // LOOP 4.4.1
                    for(int k = 0; k < 8; k++) {
                        coord_t dir_offs = neighbour_offsets[k];
                        coord_t neighbour = {
                            yes_pixel.x + dir_offs.x,
                            yes_pixel.y + dir_offs.y
                        };

                        if (neighbour.x < 0) neighbour.x = 0;
                        if (neighbour.x >= width) neighbour.x = width - 1;
                        if (neighbour.y < 0) neighbour.y = 0;
                        if (neighbour.y >= height) neighbour.y = height - 1;

                        if (image[neighbour.y * width + neighbour.x] == 127) {
                            image[neighbour.y * width + neighbour.x] = 255;

                            size_t index = num_new_yes++;
                            new_yes_pixels[index] = neighbour;

                            new_yes_found = true;
                        }
                    }
                }

                yes_pixels = new_yes_pixels;
                num_yes_pixels = num_new_yes;
            }

    } while(new_yes_found);

    // Version of the LOOP 4.3.1 but with the critical section moved inside LOOP 4.4.1
    /*
    // LOOP 4.3.1
    do {
            new_yes_found = false;

            coord_t *new_yes_pixels = malloc(width * height * sizeof(coord_t));
            size_t num_new_yes = 0;

            // LOOP 4.3.2
            #pragma omp parallel for
            for(size_t i = 0; i < num_yes_pixels; i++) {
                coord_t yes_pixel = yes_pixels[i];

                // LOOP 4.4.1
                for(int k = 0; k < 8; k++) {
                    coord_t dir_offs = neighbour_offsets[k];
                    coord_t neighbour = {
                        yes_pixel.x + dir_offs.x,
                        yes_pixel.y + dir_offs.y
                    };

                    if (neighbour.x < 0) neighbour.x = 0;
                    if (neighbour.x >= width) neighbour.x = width - 1;
                    if (neighbour.y < 0) neighbour.y = 0;
                    if (neighbour.y >= height) neighbour.y = height - 1;

                    #pragma omp critical
                    {
                        if (image[neighbour.y * width + neighbour.x] == 127) {
                            image[neighbour.y * width + neighbour.x] = 255;

                            size_t index = num_new_yes++;
                            new_yes_pixels[index] = neighbour;

                            new_yes_found = true;
                        }
                    }
                }
            }

            yes_pixels = new_yes_pixels;
            num_yes_pixels = num_new_yes;

    } while(new_yes_found);
    */

    free(yes_pixels);

    #pragma omp parallel for
    // LOOP 4.5
    for (int y = 0; y < height; y++) {
        // LOOP 4.6
        for (int x = 0; x < width; x++) {
            uint8_t value = image[y * width + x];
            bool condition = (value == 127);

            image[y * width + x] = condition ? 0 : value;
        }
    }
}

void
cannyEdgeDetection(
    uint8_t *restrict input, size_t width, size_t height,
    uint16_t threshold_lower, uint16_t threshold_upper,
    uint8_t *restrict output, double *restrict runtimes) {
    size_t image_size = width * height;

    // Allocate arrays for intermediate results
    int16_t *sobel_x = malloc(image_size * sizeof(int16_t));
    assert(sobel_x);

    int16_t *sobel_y = malloc(image_size * sizeof(int16_t));
    assert(sobel_y);

    uint8_t *phase = malloc(image_size * sizeof(uint8_t));
    assert(phase);

    uint16_t *magnitude = malloc(image_size * sizeof(uint16_t));
    assert(magnitude);

    uint64_t times[5];
    // Canny edge detection algorithm consists of the following functions:
    times[0] = gettimemono_ns();
    sobel3x3(input, width, height, sobel_x, sobel_y);

    times[1] = gettimemono_ns();
    phaseAndMagnitude(sobel_x, sobel_y, width, height, phase, magnitude);

    times[2] = gettimemono_ns();
    nonMaxSuppression(
        magnitude, phase, width, height, threshold_lower, threshold_upper,
        output);

    times[3] = gettimemono_ns();
    edgeTracing(output, width, height);  // modifies output in-place

    times[4] = gettimemono_ns();
    // Release intermediate arrays
    free(sobel_x);
    free(sobel_y);
    free(phase);
    free(magnitude);

    for (int i = 0; i < 4; i++) {
        runtimes[i] = times[i + 1] - times[i];
        runtimes[i] /= 1000000.0;  // Convert ns to ms
    }
}

// Needed only in Part 2 for OpenCL initialization
void
init(
    size_t width, size_t height, uint16_t threshold_lower,
    uint16_t threshold_upper) {}

void
destroy() {}

////////////////////////////////////////////////
// ¤¤ DO NOT EDIT ANYTHING AFTER THIS LINE ¤¤ //
////////////////////////////////////////////////

enum PROCESSING_MODE { DEFAULT, BIG_MODE, SMALL_MODE, VIDEO_MODE };
// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
int
main(int argc, char **argv) {
    enum PROCESSING_MODE mode = DEFAULT;
    if (argc > 1) {
        char *mode_c = argv[1];
        if (strlen(mode_c) == 2) {
            if (strncmp(mode_c, "-B", 2) == 0) {
                mode = BIG_MODE;
            } else if (strncmp(mode_c, "-b", 2) == 0) {
                mode = SMALL_MODE;
            } else if (strncmp(mode_c, "-v", 2) == 0) {
                mode = VIDEO_MODE;
            } else {
                printf(
                    "Invalid usage! Please set either -b, -B, -v or "
                    "nothing\n");
                return -1;
            }
        } else {
            printf("Invalid usage! Please set either -b, -B, -v nothing\n");
            return -1;
        }
    }
    int benchmarking_iterations = 1;
    if (argc > 2) {
        benchmarking_iterations = atoi(argv[2]);
    }

    char *input_image_path = "";
    char *output_image_path = "";
    uint16_t threshold_lower = 0;
    uint16_t threshold_upper = 0;
    switch (mode) {
        case BIG_MODE:
            input_image_path = "hameensilta.pgm";
            output_image_path = "hameensilta_output.pgm";
            // Arbitrarily selected to produce a nice-looking image
            // DO NOT CHANGE THESE WHEN BENCHMARKING
            threshold_lower = 120;
            threshold_upper = 300;
            printf(
                "Enabling %d benchmarking iterations with the large %s "
                "image\n",
                benchmarking_iterations, input_image_path);
            break;
        case SMALL_MODE:
            input_image_path = "x.pgm";
            output_image_path = "x_output.pgm";
            threshold_lower = 750;
            threshold_upper = 800;
            printf(
                "Enabling %d benchmarking iterations with the small %s "
                "image\n",
                benchmarking_iterations, input_image_path);
            break;
        case VIDEO_MODE:
            if (system("which ffmpeg > /dev/null 2>&1") ||
                system("which ffplay > /dev/null 2>&1")) {
                printf(
                    "Video mode is disabled because ffmpeg is not found\n");
                return -1;
            }
            benchmarking_iterations = 0;
            input_image_path = "people.mp4";
            threshold_lower = 120;
            threshold_upper = 300;
            printf(
                "Playing video %s with FFMPEG. Error check disabled.\n",
                input_image_path);
            break;
        case DEFAULT:
        default:
            input_image_path = "x.pgm";
            output_image_path = "x_output.pgm";
            // Carefully selected to produce a discontinuous edge without edge
            // tracing
            threshold_lower = 750;
            threshold_upper = 800;
            printf("Running with %s image\n", input_image_path);
            break;
    }

    uint8_t *input_image = NULL;
    size_t width = 0;
    size_t height = 0;
    if (mode == VIDEO_MODE) {
        width = 3840;
        height = 2160;
        init(width, height, threshold_lower, threshold_upper);

        uint8_t *output_image = malloc(width * height);
        assert(output_image);

        int count;
        uint8_t *frame = malloc(width * height * 3);
        assert(frame);
        char pipein_cmd[1024];
        snprintf(
            pipein_cmd, 1024,
            "ffmpeg -i %s -f image2pipe -vcodec rawvideo -an -s %zux%zu "
            "-pix_fmt gray - 2> /dev/null",
            input_image_path, width, height);
        FILE *pipein = popen(pipein_cmd, "r");
        char pipeout_cmd[1024];
        snprintf(
            pipeout_cmd, 1024,
            "ffplay -f rawvideo -pixel_format gray -video_size %zux%zu "
            "-an - 2> /dev/null",
            width, height);
        FILE *pipeout = popen(pipeout_cmd, "w");
        double runtimes[4];
        while (1) {
            count = fread(frame, 1, height * width, pipein);
            if (count != height * width) break;

            cannyEdgeDetection(
                frame, width, height, threshold_lower, threshold_upper,
                output_image, runtimes);

            double total_time =
                runtimes[0] + runtimes[1] + runtimes[2] + runtimes[3];
            printf("FPS: %0.1f\n", 1000 / total_time);
            fwrite(output_image, 1, height * width, pipeout);
        }
        fflush(pipein);
        pclose(pipein);
        fflush(pipeout);
        pclose(pipeout);
    } else {
        if ((input_image = read_pgm(input_image_path, &width, &height))) {
            printf(
                "Input image read succesfully. Size %zux%zu\n", width,
                height);
        } else {
            printf("Read failed\n");
            return -1;
        }
        init(width, height, threshold_lower, threshold_upper);

        uint8_t *output_image = malloc(width * height);
        assert(output_image);

        int all_the_runs_were_succesful = 1;
        double avg_runtimes[4] = {0.0, 0.0, 0.0, 0.0};
        double avg_total = 0.0;
        for (int iter = 0; iter < benchmarking_iterations; iter++) {
            double iter_runtimes[4];
            cannyEdgeDetection(
                input_image, width, height, threshold_lower, threshold_upper,
                output_image, iter_runtimes);

            for (int n = 0; n < 4; n++) {
                avg_runtimes[n] += iter_runtimes[n] / benchmarking_iterations;
                avg_total += iter_runtimes[n] / benchmarking_iterations;
            }

            uint8_t *output_image_ref = malloc(width * height);
            assert(output_image_ref);
            cannyEdgeDetection_ref(
                input_image, width, height, threshold_lower, threshold_upper,
                output_image_ref);

            uint8_t *fused_comparison = malloc(width * height);
            assert(fused_comparison);
            int failed = validate_result(
                output_image, output_image_ref, width, height,
                fused_comparison);
            if (failed) {
                all_the_runs_were_succesful = 0;
                printf(
                    "Error checking failed for benchmark iteration %d!\n"
                    "Writing your output to %s. The image that should've "
                    "been generated is written to ref.pgm\n"
                    "Generating fused.pgm for debugging purpose. Light-grey "
                    "pixels should've been white and "
                    "dark-grey pixels black. Corrupted pixels are colored "
                    "middle-grey\n",
                    iter, output_image_path);

                write_pgm("ref.pgm", output_image_ref, width, height);
                write_pgm("fused.pgm", fused_comparison, width, height);
            }
        }

        printf("Sobel3x3 time          : %0.3f ms\n", avg_runtimes[0]);
        printf("phaseAndMagnitude time : %0.3f ms\n", avg_runtimes[1]);
        printf("nonMaxSuppression time : %0.3f ms\n", avg_runtimes[2]);
        printf("edgeTracing time       : %0.3f ms\n", avg_runtimes[3]);
        printf("Total time             : %0.3f ms\n", avg_total);
        write_pgm(output_image_path, output_image, width, height);
        printf("Wrote output to %s\n", output_image_path);
        if (all_the_runs_were_succesful) {
            printf("Error checks passed!\n");
        } else {
            printf("There were failing runs\n");
        }
    }
    destroy();
    return 0;
}
