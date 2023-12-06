int
idx(int x, int y, int width, int height, int xoff, int yoff) {
    int resx = x;

    bool is_positive_x_off = xoff > 0;
    bool is_valid_positive_range_x = is_positive_x_off && x < (width - xoff);
    bool is_valid_negative_range_x = !is_positive_x_off && x >= -xoff;

    bool final_condition_x = (is_valid_positive_range_x || is_valid_negative_range_x);
    resx += final_condition_x ? xoff : 0;

    int resy = y;

    bool is_positive_y_off = yoff > 0;
    bool is_valid_positive_range_y = is_positive_y_off && y < (height - yoff);
    bool is_valid_negative_range_y = !is_positive_y_off && y >= -yoff;

    bool final_condition_y = is_valid_positive_range_y || is_valid_negative_range_y;
    resy += final_condition_y ? yoff : 0;

    return resy * width + resx;

}

/*
__kernel void
sobel3x3_vectorized(__global const uchar4 *in, int width, int height, __global short4 *output_x, __global short4 *output_y)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int gid = y * (width >> 2) + x;

    width = width >> 2;
    height = height >> 2;

    short4 p0 = convert_short4(in[idx(x, y, width, height, -1, -1)]);
    short4 p1 = convert_short4(in[idx(x, y, width, height, 1, -1)]);

    short4 p2 = convert_short4(in[idx(x, y, width, height, -1, 0)]);
    p2 += p2;

    short4 p3 = convert_short4(in[idx(x, y, width, height, 1, 0)]);
    p3 += p3;

    short4 p4 = convert_short4(in[idx(x, y, width, height, -1, 1)]);
    short4 p5 = convert_short4(in[idx(x, y, width, height, 1, 1)]);

    short4 p6 = convert_short4(in[idx(x, y, width, height, -1, -1)]);

    short4 p7 = convert_short4(in[idx(x, y, width, height, -1, 1)]);

    short4 p8 = convert_short4(in[idx(x, y, width, height, 0, -1)]);
    p8 += p8;

    short4 p9 = convert_short4(in[idx(x, y, width, height, 0, 1)]);
    p9 += p9;

    short4 p10 = convert_short4(in[idx(x, y, width, height, 1, -1)]);
    short4 p11 = convert_short4(in[idx(x, y, width, height, 1, 1)]);

    short4 res_x = -p0 + p1 - p2 + p3 - p4 + p5;
    short4 res_y = -p6 + p7 - p8 + p9 - p10 + p11;

    // 3x3 sobel filter, first in x direction
    output_x[gid] = res_x;

    // 3x3 sobel filter, in y direction
    output_y[gid] = res_y;
}
*/

__kernel void
sobel3x3(__global const uchar *in, const int width, const int height, __global short *output_x, __global short *output_y)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int gid = y * width + x;

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

__kernel void
phaseAndMagnitude(__global const short *in_x, __global const short *in_y, int width, int height, __global uchar *phase_out, __global ushort *magnitude_out)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int gid = y * width + x;

    // Output in range -PI:PI
    float angle = atan2pi((float)in_y[gid], (float)in_x[gid]);

    // Shift range -127.5:127.5
    angle *= 127.5;

    // Shift range 0.5:255.5
    angle += (127.5 + 0.5);

    // Downcasting truncates angle to range 0:255
    phase_out[gid] = (uchar)angle;

    magnitude_out[gid] = abs(in_x[gid]) + abs(in_y[gid]);
}

__kernel void
phaseAndMagnitudeVec(__global const short4 *in_x,
                     __global const short4 *in_y,
                     int width, int height,
                     __global uchar4 *phase_out,
                     __global ushort4 *magnitude_out)
{

    int x = get_global_id(0);
    int y = get_global_id(1);
    int gid = y * (width >> 2) + x;

    short4 val_x = in_x[gid];
    short4 val_y = in_y[gid];

    float4 angle = atan2pi(convert_float4(val_y), convert_float4(val_x));
    angle *= (float4)127.5;
    angle += ((float4)127.5f + (float4)0.5f);

    phase_out[gid] = convert_uchar4(angle);
    magnitude_out[gid] = abs(val_x) + abs(val_y);
}

__kernel void
non_Max_Suppression(__global const ushort *magnitude, __global const uchar *phase, int width, int height, const ushort threshold_lower, const ushort threshold_upper, __global uchar *out)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int gid = y * width + x;

    uchar sobel_angle = phase[gid];

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

    ushort sobel_magnitude = magnitude[gid];

    // Non-maximum suppression
    // Pick out the two neighbours that are perpendicular to the current edge pixel
    ushort neighbour_max = 0;
    ushort neighbour_max2 = 0;
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

    // Double thresholding
    // Marks YES pixels with 255, NO pixels with 0 and MAYBE pixels
    // with 127
    uchar t = 127;
    if (sobel_magnitude > threshold_upper) t = 255;
    if (sobel_magnitude <= threshold_lower) t = 0;
    out[gid] = (ushort)t;
}