#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <cstring>
#include "../utils.hpp"

// Helper function to clamp pixel values between 0 and 255
inline unsigned char clamp_pixel_value(float value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return static_cast<unsigned char>(value);
}

// SIMD optimized exp function
__m256 _mm256_exp_ps(__m256 invec) {
    float *element = (float *)&invec;
    return _mm256_setr_ps(
        expf(element[0]),
        expf(element[1]),
        expf(element[2]),
        expf(element[3]),
        expf(element[4]),
        expf(element[5]),
        expf(element[6]),
        expf(element[7])
    );
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Read JPEG File using utils.cpp
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Filtered image buffer
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    memset(filteredImage, 0, input_jpeg.width * input_jpeg.height * input_jpeg.num_channels);

    // Constants for bilateral filter
    float sigma_s = 15.0f; // Spatial kernel standard deviation
    float sigma_r = 30.0f; // Range kernel standard deviation
    float inv_sigma_s = 1.0f / (2 * sigma_s * sigma_s);
    float inv_sigma_r = 1.0f / (2 * sigma_r * sigma_r);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform SIMD-optimized bilateral filtering
    for (int y = 1; y < input_jpeg.height - 1; y++) {
        for (int x = 1; x < input_jpeg.width - 1; x++) {
            int r_id = (y * input_jpeg.width + x) * input_jpeg.num_channels;
            int g_id = r_id + 1;
            int b_id = r_id + 2;

            float r_sum = 0, g_sum = 0, b_sum = 0;
            float norm_factor = 0;

            // Precompute spatial weights for 3x3 kernel using SIMD
            __m256 spatial_weights[3][3];
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    __m256 spatial_dist = _mm256_set1_ps(kx * kx + ky * ky);
                    spatial_weights[ky + 1][kx + 1] = _mm256_exp_ps(_mm256_mul_ps(spatial_dist, _mm256_set1_ps(-inv_sigma_s)));
                }
            }

            // Iterate over the 3x3 kernel
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int neighbor_x = x + kx;
                    int neighbor_y = y + ky;
                    int neighbor_r_id = (neighbor_y * input_jpeg.width + neighbor_x) * input_jpeg.num_channels;

                    // Compute range weights for each channel using SIMD
                    __m256 range_dist_r = _mm256_set1_ps(input_jpeg.buffer[r_id] - input_jpeg.buffer[neighbor_r_id]);
                    __m256 range_weight_r = _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(range_dist_r, range_dist_r), _mm256_set1_ps(-inv_sigma_r)));

                    __m256 range_dist_g = _mm256_set1_ps(input_jpeg.buffer[g_id] - input_jpeg.buffer[neighbor_r_id + 1]);
                    __m256 range_weight_g = _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(range_dist_g, range_dist_g), _mm256_set1_ps(-inv_sigma_r)));

                    __m256 range_dist_b = _mm256_set1_ps(input_jpeg.buffer[b_id] - input_jpeg.buffer[neighbor_r_id + 2]);
                    __m256 range_weight_b = _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(range_dist_b, range_dist_b), _mm256_set1_ps(-inv_sigma_r)));

                    // Multiply spatial and range weights
                    __m256 weight_r = _mm256_mul_ps(spatial_weights[ky + 1][kx + 1], range_weight_r);
                    __m256 weight_g = _mm256_mul_ps(spatial_weights[ky + 1][kx + 1], range_weight_g);
                    __m256 weight_b = _mm256_mul_ps(spatial_weights[ky + 1][kx + 1], range_weight_b);

                    // Sum weighted pixel values
                    r_sum += input_jpeg.buffer[neighbor_r_id] * weight_r[0];
                    g_sum += input_jpeg.buffer[neighbor_r_id + 1] * weight_g[0];
                    b_sum += input_jpeg.buffer[neighbor_r_id + 2] * weight_b[0];

                    // Accumulate normalization factor
                    norm_factor += weight_r[0] + weight_g[0] + weight_b[0];
                }
            }

            // Normalize the results and clamp the values
            filteredImage[r_id] = clamp_pixel_value(r_sum / norm_factor);
            filteredImage[g_id] = clamp_pixel_value(g_sum / norm_factor);
            filteredImage[b_id] = clamp_pixel_value(b_sum / norm_factor);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output JPEG image using utils.cpp
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Post-processing
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}