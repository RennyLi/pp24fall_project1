//
// Created by Liu Yuxuan on 2024/9/10
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Row-wise Pthread parallel implementation of smooth image filtering of JPEG
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <pthread.h>
#include <cmath>
#include "../utils.hpp"

// create a structure to pass data to individual threads
struct ThreadData {
    JpegSOA* input_jpeg;
    JpegSOA* output_jpeg;
    int width;
    int height;
    int start_row;
    int end_row;
    float sigma_s;
    float sigma_r;
};

inline unsigned char clamp_pixel_value(float value) {
    return static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, value)));
}

// bilateral filtering function executed by each thread
void* bilateral_filter_thread_function(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    for (int y = data->start_row; y < data->end_row; y++) {
        for (int x = 1; x < data->width - 1; x++) {
            int index = y * data->width + x;

            float r_sum = 0, g_sum = 0, b_sum = 0;
            float norm_factor_r = 0, norm_factor_g = 0, norm_factor_b = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int neighbor_x = x + kx;
                    int neighbor_y = y + ky;
                    int neighbor_index = neighbor_y * data->width + neighbor_x;

                    float spatial_dist = kx * kx + ky * ky;
                    float spatial_weight = expf(-spatial_dist / (2 * data->sigma_s * data->sigma_s));

                    float range_dist_r = data->input_jpeg->r_values[index] - data->input_jpeg->r_values[neighbor_index];
                    float range_weight_r = expf(-(range_dist_r * range_dist_r) / (2 * data->sigma_r * data->sigma_r));
                    float weight_r = spatial_weight * range_weight_r;
                    r_sum += data->input_jpeg->r_values[neighbor_index] * weight_r;
                    norm_factor_r += weight_r;

                    float range_dist_g = data->input_jpeg->g_values[index] - data->input_jpeg->g_values[neighbor_index];
                    float range_weight_g = expf(-(range_dist_g * range_dist_g) / (2 * data->sigma_r * data->sigma_r));
                    float weight_g = spatial_weight * range_weight_g;
                    g_sum += data->input_jpeg->g_values[neighbor_index] * weight_g;
                    norm_factor_g += weight_g;

                    float range_dist_b = data->input_jpeg->b_values[index] - data->input_jpeg->b_values[neighbor_index];
                    float range_weight_b = expf(-(range_dist_b * range_dist_b) / (2 * data->sigma_r * data->sigma_r));
                    float weight_b = spatial_weight * range_weight_b;
                    b_sum += data->input_jpeg->b_values[neighbor_index] * weight_b;
                    norm_factor_b += weight_b;
                }
            }

            data->output_jpeg->r_values[index] = clamp_pixel_value(r_sum / norm_factor_r);
            data->output_jpeg->g_values[index] = clamp_pixel_value(g_sum / norm_factor_g);
            data->output_jpeg->b_values[index] = clamp_pixel_value(b_sum / norm_factor_b);
        }
    }

    return nullptr;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
        return -1;
    }
    
    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    int NUM_THREADS = std::stoi(argv[3]);

    std::cout << "Input file from: " << input_filename << "\n";
    
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // prepare output image
    JpegSOA output_jpeg;
    output_jpeg.width = input_jpeg.width;
    output_jpeg.height = input_jpeg.height;
    output_jpeg.num_channels = input_jpeg.num_channels;
    output_jpeg.color_space = input_jpeg.color_space;

    output_jpeg.r_values = new ColorValue[output_jpeg.width * output_jpeg.height];
    output_jpeg.g_values = new ColorValue[output_jpeg.width * output_jpeg.height];
    output_jpeg.b_values = new ColorValue[output_jpeg.width * output_jpeg.height];

    float sigma_s = 15.0f; 
    float sigma_r = 30.0f; 

    // create threads
    pthread_t* threads = new pthread_t[NUM_THREADS];
    ThreadData* threadData = new ThreadData[NUM_THREADS];
    int rowsPerThread = input_jpeg.height / NUM_THREADS;

    auto start_time = std::chrono::high_resolution_clock::now(); 

    for (int i = 0; i < NUM_THREADS; i++) {
        threadData[i] = {&input_jpeg,
                         &output_jpeg,
                         input_jpeg.width,
                         input_jpeg.height,
                         i * rowsPerThread,
                         (i == NUM_THREADS - 1) ? input_jpeg.height : (i + 1) * rowsPerThread,
                         sigma_s,
                         sigma_r};
        pthread_create(&threads[i], NULL, bilateral_filter_thread_function, &threadData[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    auto end_time = std::chrono::high_resolution_clock::now(); 

    std::cout << "Output file to: " << output_filename << "\n";
    if (export_jpeg(output_jpeg, output_filename)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    delete[] input_jpeg.r_values;
    delete[] input_jpeg.g_values;
    delete[] input_jpeg.b_values;
    delete[] output_jpeg.r_values;
    delete[] output_jpeg.g_values;
    delete[] output_jpeg.b_values;
    delete[] threads;
    delete[] threadData;

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}