//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// OpenACC implementation of bilateral filtering on JPEG image
//

#include <memory.h>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iostream>
#include <openacc.h>

#include "../utils.hpp"

#pragma acc routine seq
ColorValue acc_clamp_pixel_value(float value)
{
    return value > 255 ? 255
           : value < 0 ? 0
                       : static_cast<unsigned char>(value);
}

#pragma acc routine seq
float acc_bilateral_filter(const ColorValue* values, int row, int col, int width,
                           float sigma_s, float sigma_r)
{
    float w_spatial[3][3] = {0};
    float w_intensity[3][3] = {0};
    float value_center = values[row * width + col];
    float sum_weights = 0.0f;
    float filtered_value = 0.0f;

    for (int ky = -1; ky <= 1; ++ky)
    {
        for (int kx = -1; kx <= 1; ++kx)
        {
            int idx_y = row + ky;
            int idx_x = col + kx;
            int idx = idx_y * width + idx_x;
            float spatial_dist = ky * ky + kx * kx;
            float intensity_dist = value_center - values[idx];

            w_spatial[ky + 1][kx + 1] = expf(-spatial_dist / (2 * sigma_s * sigma_s));
            w_intensity[ky + 1][kx + 1] = expf(-intensity_dist * intensity_dist / (2 * sigma_r * sigma_r));

            float weight = w_spatial[ky + 1][kx + 1] * w_intensity[ky + 1][kx + 1];
            filtered_value += values[idx] * weight;
            sum_weights += weight;
        }
    }
    return acc_clamp_pixel_value(filtered_value / sum_weights);
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Read JPEG File
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Constants for bilateral filter
    float sigma_s = 15.0f;
    float sigma_r = 30.0f;
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    size_t image_size = width * height * sizeof(ColorValue);

    // Allocate output arrays
    ColorValue* filtered_r = new ColorValue[width * height];
    ColorValue* filtered_g = new ColorValue[width * height];
    ColorValue* filtered_b = new ColorValue[width * height];

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Explicitly move the structure member arrays to the GPU, instead of the whole structure
#pragma acc enter data copyin(input_jpeg.r_values[0:width*height], \
                              input_jpeg.g_values[0:width*height], \
                              input_jpeg.b_values[0:width*height], \
                              filtered_r[0:width*height], \
                              filtered_g[0:width*height], \
                              filtered_b[0:width*height])

#pragma acc parallel present(input_jpeg.r_values, input_jpeg.g_values, input_jpeg.b_values, \
                             filtered_r, filtered_g, filtered_b) \
    num_gangs(1024) vector_length(256)
    {
#pragma acc loop independent
        for (int y = 1; y < height - 1; ++y)
        {
#pragma acc loop independent
            for (int x = 1; x < width - 1; ++x)
            {
                filtered_r[y * width + x] = acc_bilateral_filter(input_jpeg.r_values, y, x, width, sigma_s, sigma_r);
                filtered_g[y * width + x] = acc_bilateral_filter(input_jpeg.g_values, y, x, width, sigma_s, sigma_r);
                filtered_b[y * width + x] = acc_bilateral_filter(input_jpeg.b_values, y, x, width, sigma_s, sigma_r);
            }
        }
    }

#pragma acc update self(filtered_r[0:width*height], \
                        filtered_g[0:width*height], \
                        filtered_b[0:width*height])

#pragma acc exit data delete(input_jpeg.r_values, input_jpeg.g_values, input_jpeg.b_values, \
                             filtered_r, filtered_g, filtered_b)

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JpegSOA output_jpeg{filtered_r, filtered_g, filtered_b, width, height, input_jpeg.num_channels, input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Post-processing
    delete[] filtered_r;
    delete[] filtered_g;
    delete[] filtered_b;
    std::cout << "Transformation Complete!" << std::endl;

    return 0;
}