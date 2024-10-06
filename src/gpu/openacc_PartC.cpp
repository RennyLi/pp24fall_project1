//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// OpenACC implementation of image filtering on JPEG
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
float acc_bilateral_filter(const unsigned char* image_buffer, int pixel_id, int width, int num_channels,
                           float sigma_s, float sigma_r, int x, int y, int channel_id) {
    int line_width = width * num_channels;
    float sum_weights = 0.0f;
    float filtered_value = 0.0f;

    float central_value = image_buffer[pixel_id];
    float sigma_s_sq = 2.0f * sigma_s * sigma_s;
    float sigma_r_sq = 2.0f * sigma_r * sigma_r;

    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int neighbor_x = x + kx;
            int neighbor_y = y + ky;

            // boudary handling, ensure no out-of-bounds pixel access
            if (neighbor_x < 0 || neighbor_x >= width || neighbor_y < 0 || neighbor_y >= width) {
                continue;
            }

            int neighbor_id = (neighbor_y * width + neighbor_x) * num_channels + channel_id;
            float neighbor_value = image_buffer[neighbor_id];

            float spatial_dist = kx * kx + ky * ky;
            float w_spatial = expf(-spatial_dist / sigma_s_sq);

            float intensity_dist = central_value - neighbor_value;
            float w_intensity = expf(-(intensity_dist * intensity_dist) / sigma_r_sq);

            float weight = w_spatial * w_intensity;

            filtered_value += neighbor_value * weight;
            sum_weights += weight;
        }
    }

    // return the normalized filtering result
    return filtered_value / sum_weights;
}

int main(int argc, char** argv) {
    if (argc != 3) {
      	std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);

    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    size_t buffer_size = width * height * num_channels;
    unsigned char* filteredImage = new unsigned char[buffer_size];
    unsigned char* buffer = new unsigned char[buffer_size];

    memcpy(buffer, input_jpeg.buffer, buffer_size);
    delete[] input_jpeg.buffer;

    float sigma_s = 15.0f;  // spatial
    float sigma_r = 30.0f;  // range

#pragma acc enter data copyin(  filteredImage[0: buffer_size],  \
                                buffer[0: buffer_size],     \
                                sigma_s, sigma_r)
#pragma acc update device(  filteredImage[0: buffer_size],  \
                            buffer[0: buffer_size])
    auto start_time = std::chrono::high_resolution_clock::now();

    // apply bilateral filtering on each color channel
#pragma acc parallel present(filteredImage[0:buffer_size], buffer[0:buffer_size], sigma_s, sigma_r) num_gangs(1024)
    {
#pragma acc loop independent
        for (int y = 1; y < height - 1; y++) {
#pragma acc loop independent
            for (int x = 1; x < width - 1; x++) {
                int r_id = (y * width + x) * num_channels;
                int g_id = r_id + 1;
                int b_id = r_id + 2;

                float r = acc_bilateral_filter(buffer, r_id, width, num_channels, sigma_s, sigma_r, x, y, 0);
                float g = acc_bilateral_filter(buffer, g_id, width, num_channels, sigma_s, sigma_r, x, y, 1);
                float b = acc_bilateral_filter(buffer, b_id, width, num_channels, sigma_s, sigma_r, x, y, 2);

                filteredImage[r_id] = acc_clamp_pixel_value(r);
                filteredImage[g_id] = acc_clamp_pixel_value(g);
                filteredImage[b_id] = acc_clamp_pixel_value(b);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

#pragma acc update self(filteredImage[0: buffer_size])
#pragma acc exit data copyout(filteredImage[0 : buffer_size])

    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    delete[] buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}
