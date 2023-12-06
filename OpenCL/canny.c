/* COMP.CE.350 Parallelization Exercise 2023
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
#include "opencl_util.h"

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
// Source code
static char *source_code;

size_t datasize_out_sobel;
size_t datasize_out_magnitude;

// Variables for OpenCL boiler code
static cl_platform_id *platforms = NULL;
static cl_device_id *devices;
static cl_context context;
static cl_command_queue cmdQueue;
static cl_program program;

// Kernels
static cl_kernel kernel_sobel3x3;
static cl_kernel kernel_phase_and_magnitude;
static cl_kernel kernel_non_max_suppression;

cl_uint numDevices = 0;

// Device buffers for sobel3x3
static cl_mem buffer_input;
static cl_mem buffer_sobel_out_x;
static cl_mem buffer_sobel_out_y;

// Device buffers for phase_and_magnitude
static cl_mem buffer_phase_out;
static cl_mem buffer_magnitude_out;

// Device buffers for non_max_suppression
static cl_mem buffer_output;

// Events Sobel3x3
static cl_event kernel_event_sobel3x3;
static cl_event write_out_x_event_sobel;
static cl_event write_out_y_event_sobel;

// Events Phase_and_Magnitude
static cl_event kernel_event_phase_and_magnitude;
static cl_event out_event_phase;
static cl_event out_event_magnitude;

// Events non_Max_Suppression
static cl_event kernel_event_non_max_suppression;

// Events for profiling
static cl_event input_image_write_event;
static cl_event output_image_read_event;

// Check output of each API call
cl_int status;

// ##########################################

// Utility function to convert 2d index with offset to linear index
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

void ErrorCheck(const char *status) {
    if(strcmp(status, "Success!") != 0) {
        printf("ERROR! %s\n", status);
        exit(1);
    }
}

void
cannyEdgeDetection(
    uint8_t *restrict input, size_t width, size_t height,
    uint16_t threshold_lower, uint16_t threshold_upper,
    uint8_t *restrict output, double *restrict runtimes) {

    size_t image_size = width * height;
    width = (int)width;
    height = (int)height;

    // Enqueue input buffer
    status = clEnqueueWriteBuffer(cmdQueue, buffer_input, CL_FALSE, 0, image_size, input, 0, NULL, &input_image_write_event);
    ErrorCheck(clErrorString(status));

    // Create a program
    program = clCreateProgramWithSource(context, 1, (const char**)& source_code, NULL, &status);
    ErrorCheck(clErrorString(status));

    // Build (compile) the program for the device
    status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
    ErrorCheck(clErrorString(status));

    // Define an index space (global work size) of work items for execution
    size_t globalWorkSize[2];
    globalWorkSize[0] = width;
    globalWorkSize[1] = height;

    // Global work size for phaseAndMagnitude kernel
    size_t globalWorkSizePhaseAndMagnitude[2];
    globalWorkSizePhaseAndMagnitude[0] = width / 4;
    globalWorkSizePhaseAndMagnitude[1] = height;

    /*
    size_t globalWorkSizeNonMax[2];
    globalWorkSizeSobel[0] = width / 4;
    globalWorkSizeSobel[1] = height / 4;
    */

    // ################################################
    // Sobel3x3
    // Create kernel for sobel3x3
    kernel_sobel3x3 = clCreateKernel(program, "sobel3x3", &status);
    ErrorCheck(clErrorString(status));

    // Associate the input and output buffers with the kernel
    status = clSetKernelArg(kernel_sobel3x3, 0, sizeof(cl_mem), &buffer_input);
    status = clSetKernelArg(kernel_sobel3x3, 1, sizeof(int), &width);
    status = clSetKernelArg(kernel_sobel3x3, 2, sizeof(int), &height);
    status = clSetKernelArg(kernel_sobel3x3, 3, sizeof(cl_mem), &buffer_sobel_out_x);
    status = clSetKernelArg(kernel_sobel3x3, 4, sizeof(cl_mem), &buffer_sobel_out_y);
    ErrorCheck(clErrorString(status));

    // Execute the kernel for sobel3x3
    status = clEnqueueNDRangeKernel(cmdQueue, kernel_sobel3x3, 2, NULL, globalWorkSize, NULL, 0, NULL, &kernel_event_sobel3x3);
    ErrorCheck(clErrorString(status));
    // ################################################

    // ################################################
    // Phase_and_Magnitude
    kernel_phase_and_magnitude = clCreateKernel(program, "phaseAndMagnitudeVec", &status);
    ErrorCheck(clErrorString(status));

    status = clSetKernelArg(kernel_phase_and_magnitude, 0, sizeof(cl_mem), &buffer_sobel_out_x);
    status = clSetKernelArg(kernel_phase_and_magnitude, 1, sizeof(cl_mem), &buffer_sobel_out_y);
    status = clSetKernelArg(kernel_phase_and_magnitude, 2, sizeof(int), &width);
    status = clSetKernelArg(kernel_phase_and_magnitude, 3, sizeof(int), &height);
    status = clSetKernelArg(kernel_phase_and_magnitude, 4, sizeof(cl_mem), &buffer_phase_out);
    status = clSetKernelArg(kernel_phase_and_magnitude, 5, sizeof(cl_mem), &buffer_magnitude_out);
    ErrorCheck(clErrorString(status));

    status = clEnqueueNDRangeKernel(cmdQueue, kernel_phase_and_magnitude, 2, NULL, globalWorkSizePhaseAndMagnitude, NULL, 0, NULL, &kernel_event_phase_and_magnitude);
    ErrorCheck(clErrorString(status));
    // ################################################

    // ################################################
    // non_Max_Suppression
    kernel_non_max_suppression = clCreateKernel(program, "non_Max_Suppression", &status);
    ErrorCheck(clErrorString(status));

    status = clSetKernelArg(kernel_non_max_suppression, 0, sizeof(cl_mem), &buffer_magnitude_out);
    status = clSetKernelArg(kernel_non_max_suppression, 1, sizeof(cl_mem), &buffer_phase_out);
    status = clSetKernelArg(kernel_non_max_suppression, 2, sizeof(int), &width);
    status = clSetKernelArg(kernel_non_max_suppression, 3, sizeof(int), &height);
    status = clSetKernelArg(kernel_non_max_suppression, 4, sizeof(uint16_t), &threshold_lower);
    status = clSetKernelArg(kernel_non_max_suppression, 5, sizeof(uint16_t), &threshold_upper);
    status = clSetKernelArg(kernel_non_max_suppression, 6, sizeof(cl_mem), &buffer_output);
    ErrorCheck(clErrorString(status));

    status = clEnqueueNDRangeKernel(cmdQueue, kernel_non_max_suppression, 2, NULL, globalWorkSize, NULL, 0, NULL, &kernel_event_non_max_suppression);
    ErrorCheck(clErrorString(status));

    clEnqueueReadBuffer(cmdQueue, buffer_output, CL_TRUE, 0, image_size, output, 0, NULL, &output_image_read_event);
    // ################################################

    uint64_t times[2];
    times[0] = gettimemono_ns();
    edgeTracing(output, width, height);
    times[1] = gettimemono_ns();

    // Getting kernels execution times
    double sobel_time = (double) getStartEndTime(kernel_event_sobel3x3) / 1000000.0;
    double phase_magnitude_time = (double) getStartEndTime(kernel_event_phase_and_magnitude) / 1000000.0;
    double non_max_suppression_time = (double) getStartEndTime(kernel_event_non_max_suppression) / 1000000.0;
    double input_image_transfer_time = (double) getStartEndTime(input_image_write_event) / 1000000.0;
    double output_image_transfer_time = (double) getStartEndTime(output_image_read_event) / 1000000.0;

    printf("input transfer time: %.6lf ms\n", input_image_transfer_time);
    printf("output transfer time: %.6lf ms\n", output_image_transfer_time);

    // Adding kernel times into runtimes array
    runtimes[0] = sobel_time;
    runtimes[1] = phase_magnitude_time;
    runtimes[2] = non_max_suppression_time;
    runtimes[3] = (times[1] - times[0]) / 1000000.0;
}

// Needed only in Part 2 for OpenCL initialization
void
init(
    size_t width, size_t height, uint16_t threshold_lower,
    uint16_t threshold_upper) {

    // Retrieve number of platforms
    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    ErrorCheck(clErrorString(status));

    // Allocate enough space for each platform
    platforms = (cl_platform_id *) malloc(numPlatforms * sizeof(cl_platform_id));

    // Fill in the platforms
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    ErrorCheck(clErrorString(status));

    // Retrieve the number of devices
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    ErrorCheck(clErrorString(status));

    // Allocate enough space for each device
    devices = (cl_device_id *) malloc(numDevices * sizeof(cl_device_id));

    // Fill in the devices
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
    ErrorCheck(clErrorString(status));

    // Create a context and associate it with the devices
    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
    ErrorCheck(clErrorString(status));

    // Create a command queue and associate it with the device
    cmdQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
    ErrorCheck(clErrorString(status));

    const int image_size = width * height;
    datasize_out_sobel = sizeof(int16_t) * image_size;

    // ################################################
    // Sobel3x3

    // Allocate space for input/output data
    buffer_input = clCreateBuffer(context, CL_MEM_READ_ONLY, image_size, NULL, &status);
    ErrorCheck(clErrorString(status));

    buffer_sobel_out_x = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize_out_sobel, NULL, &status);
    ErrorCheck(clErrorString(status));

    buffer_sobel_out_y = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize_out_sobel, NULL, &status);
    ErrorCheck(clErrorString(status));
    // ################################################

    // ################################################
    // Phase_and_Magnitude

    datasize_out_magnitude = sizeof(uint16_t) * image_size;

    // Allocate space for input/output data
    buffer_phase_out = clCreateBuffer(context, CL_MEM_READ_WRITE, image_size, NULL, &status);
    ErrorCheck(clErrorString(status));

    buffer_magnitude_out = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize_out_magnitude, NULL, &status);
    ErrorCheck(clErrorString(status));
    // ################################################

    // ################################################
    // non_Max_Suppression

    // Allocate space for input/output data
    buffer_output = clCreateBuffer(context, CL_MEM_READ_WRITE, image_size, NULL, &status);
    ErrorCheck(clErrorString(status));

    // ################################################

    // Reading source code only once
    source_code = read_source("canny.cl");
}

void
destroy() {

    // Free OpenCL resources
    clReleaseKernel(kernel_sobel3x3);
    clReleaseKernel(kernel_phase_and_magnitude);
    clReleaseKernel(kernel_non_max_suppression);
    clReleaseCommandQueue(cmdQueue);
    clReleaseProgram(program);
    clReleaseMemObject(buffer_input);
    clReleaseMemObject(buffer_sobel_out_x);
    clReleaseMemObject(buffer_sobel_out_y);
    clReleaseMemObject(buffer_phase_out);
    clReleaseMemObject(buffer_magnitude_out);
    clReleaseMemObject(buffer_output);
    clReleaseContext(context);

    // Free host resources
    free(platforms);
    free(devices);
}

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
