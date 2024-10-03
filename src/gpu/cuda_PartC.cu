//
// Created by Liu Yuxuan on 2024/9/11
// Modified from Zhong Yebin's PartB on 2023/9/16
//
// Email: yebinzhong@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// CUDA implementation of bilateral filtering on JPEG image
//

#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include "../utils.hpp"

// Constants for bilateral filter
__constant__ float d_sigma_s = 15.0f;
__constant__ float d_sigma_r = 30.0f;

__device__ unsigned char d_clamp_pixel_value(float pixel)
{
    return pixel > 255 ? 255
           : pixel < 0 ? 0
                       : static_cast<unsigned char>(pixel);
}

/**
 * Kernel function to perform bilateral filtering on an image channel
 */
__global__ void bilateral_filter_kernel(const ColorValue* input_channel, ColorValue* output_channel,
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int idx = y * width + x;

        float sum = 0.0f;
        float norm_factor = 0.0f;
        float center_value = input_channel[idx];

        // Iterate over the 3x3 kernel
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int neighbor_x = x + kx;
                int neighbor_y = y + ky;
                int neighbor_idx = neighbor_y * width + neighbor_x;

                float spatial_dist = kx * kx + ky * ky;
                float spatial_weight = expf(-spatial_dist / (2 * d_sigma_s * d_sigma_s));

                float range_dist = center_value - input_channel[neighbor_idx];
                float range_weight = expf(-(range_dist * range_dist) / (2 * d_sigma_r * d_sigma_r));

                float weight = spatial_weight * range_weight;
                sum += input_channel[neighbor_idx] * weight;
                norm_factor += weight;
            }
        }

        output_channel[idx] = d_clamp_pixel_value(sum / norm_factor);
    }
}

/**
 * Helper function to launch CUDA kernel
 */
void apply_bilateral_filter_cuda(const JpegSOA& input_jpeg, JpegSOA& output_jpeg, int width, int height) {
    // Allocate memory on the GPU for input and output channels
    ColorValue *d_r_input, *d_g_input, *d_b_input;
    ColorValue *d_r_output, *d_g_output, *d_b_output;

    size_t channel_size = width * height * sizeof(ColorValue);
    cudaMalloc((void**)&d_r_input, channel_size);
    cudaMalloc((void**)&d_g_input, channel_size);
    cudaMalloc((void**)&d_b_input, channel_size);
    cudaMalloc((void**)&d_r_output, channel_size);
    cudaMalloc((void**)&d_g_output, channel_size);
    cudaMalloc((void**)&d_b_output, channel_size);

    // Copy input data from host to device
    cudaMemcpy(d_r_input, input_jpeg.r_values, channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_input, input_jpeg.g_values, channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_input, input_jpeg.b_values, channel_size, cudaMemcpyHostToDevice);

    // Set CUDA grid and block dimensions
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel for each channel (R, G, B)
    bilateral_filter_kernel<<<gridDim, blockDim>>>(d_r_input, d_r_output, width, height);
    bilateral_filter_kernel<<<gridDim, blockDim>>>(d_g_input, d_g_output, width, height);
    bilateral_filter_kernel<<<gridDim, blockDim>>>(d_b_input, d_b_output, width, height);

    // Copy output data back to host
    cudaMemcpy(output_jpeg.r_values, d_r_output, channel_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_jpeg.g_values, d_g_output, channel_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_jpeg.b_values, d_b_output, channel_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_r_input);
    cudaFree(d_g_input);
    cudaFree(d_b_input);
    cudaFree(d_r_output);
    cudaFree(d_g_output);
    cudaFree(d_b_output);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Read input JPEG image in structure-of-array form
    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    std::cout << "Input file from: " << input_filename << "\n";

    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Allocate memory for output image
    JpegSOA output_jpeg;
    output_jpeg.width = input_jpeg.width;
    output_jpeg.height = input_jpeg.height;
    output_jpeg.num_channels = input_jpeg.num_channels;
    output_jpeg.color_space = input_jpeg.color_space;
    output_jpeg.r_values = new ColorValue[output_jpeg.width * output_jpeg.height];
    output_jpeg.g_values = new ColorValue[output_jpeg.width * output_jpeg.height];
    output_jpeg.b_values = new ColorValue[output_jpeg.width * output_jpeg.height];

    // Start timer
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);  // GPU start time

    // Apply bilateral filter using CUDA
    apply_bilateral_filter_cuda(input_jpeg, output_jpeg, input_jpeg.width, input_jpeg.height);

    // End timer
    cudaEventRecord(stop, 0);  // GPU end time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuDuration, start, stop);
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;

    // Save output JPEG image
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

    // Cleanup CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}