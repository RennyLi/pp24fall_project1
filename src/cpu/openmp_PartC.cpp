//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// OpenMP implementation of bilateral filtering of a JPEG image
//

#include <memory.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>

#include "../utils.hpp"

// Helper function to clamp pixel values between 0 and 255
inline unsigned char clamp_pixel_value(float value) {
    return static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, value)));
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
        return -1;
    }

    // Read JPEG File
    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    int NUM_THREADS = std::stoi(argv[3]);
    std::cout << "Input file from: " << input_filename << "\n";

    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Prepare output image
    JpegSOA output_jpeg;
    output_jpeg.width = input_jpeg.width;
    output_jpeg.height = input_jpeg.height;
    output_jpeg.num_channels = input_jpeg.num_channels;
    output_jpeg.color_space = input_jpeg.color_space;

    output_jpeg.r_values = new ColorValue[output_jpeg.width * output_jpeg.height];
    output_jpeg.g_values = new ColorValue[output_jpeg.width * output_jpeg.height];
    output_jpeg.b_values = new ColorValue[output_jpeg.width * output_jpeg.height];

    // Bilateral filter constants
    float sigma_s = 15.0f;  // Spatial kernel standard deviation
    float sigma_r = 30.0f;  // Range kernel standard deviation

    auto start_time = std::chrono::high_resolution_clock::now();  // Start time recording

    // Set the number of threads
    omp_set_num_threads(NUM_THREADS);

    // Perform bilateral filtering using OpenMP
#pragma omp parallel for
    for (int y = 1; y < input_jpeg.height - 1; y++) {
        for (int x = 1; x < input_jpeg.width - 1; x++) {
            int index = y * input_jpeg.width + x;

            float r_sum = 0, g_sum = 0, b_sum = 0;
            float norm_factor_r = 0, norm_factor_g = 0, norm_factor_b = 0;

            // Iterate over the 3x3 kernel
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int neighbor_x = x + kx;
                    int neighbor_y = y + ky;
                    int neighbor_index = neighbor_y * input_jpeg.width + neighbor_x;

                    // Compute spatial weights
                    float spatial_dist = kx * kx + ky * ky;
                    float spatial_weight = expf(-spatial_dist / (2 * sigma_s * sigma_s));

                    // Compute range weights and apply bilateral filter for each channel (R, G, B)
                    float range_dist_r = input_jpeg.r_values[index] - input_jpeg.r_values[neighbor_index];
                    float range_weight_r = expf(-(range_dist_r * range_dist_r) / (2 * sigma_r * sigma_r));
                    float weight_r = spatial_weight * range_weight_r;
                    r_sum += input_jpeg.r_values[neighbor_index] * weight_r;
                    norm_factor_r += weight_r;

                    float range_dist_g = input_jpeg.g_values[index] - input_jpeg.g_values[neighbor_index];
                    float range_weight_g = expf(-(range_dist_g * range_dist_g) / (2 * sigma_r * sigma_r));
                    float weight_g = spatial_weight * range_weight_g;
                    g_sum += input_jpeg.g_values[neighbor_index] * weight_g;
                    norm_factor_g += weight_g;

                    float range_dist_b = input_jpeg.b_values[index] - input_jpeg.b_values[neighbor_index];
                    float range_weight_b = expf(-(range_dist_b * range_dist_b) / (2 * sigma_r * sigma_r));
                    float weight_b = spatial_weight * range_weight_b;
                    b_sum += input_jpeg.b_values[neighbor_index] * weight_b;
                    norm_factor_b += weight_b;
                }
            }

            // Normalize and clamp the results for each channel
            output_jpeg.r_values[index] = clamp_pixel_value(r_sum / norm_factor_r);
            output_jpeg.g_values[index] = clamp_pixel_value(g_sum / norm_factor_g);
            output_jpeg.b_values[index] = clamp_pixel_value(b_sum / norm_factor_b);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();  // End time recording

    // Save the filtered image
    std::cout << "Output file to: " << output_filename << "\n";
    if (export_jpeg(output_jpeg, output_filename)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Cleanup
    delete[] input_jpeg.r_values;
    delete[] input_jpeg.g_values;
    delete[] input_jpeg.b_values;
    delete[] output_jpeg.r_values;
    delete[] output_jpeg.g_values;
    delete[] output_jpeg.b_values;

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}