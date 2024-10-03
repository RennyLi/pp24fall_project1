//
// Created by Yang Yufan on 2023/9/16.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI implementation of bilateral filtering on JPEG images
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <mpi.h> // MPI Header

#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

// Function to apply bilateral filter on a part of the image (parallelized using MPI)
void bilateral_filter_mpi(unsigned char* filtered_r, const unsigned char* input_r, const unsigned char* input_g, const unsigned char* input_b, 
                          int width, int height, int start_line, int end_line, int radius, float sigma_s, float sigma_r)
{
    // Precompute some constants
    const float inv_two_sigma_s2 = 1.0f / (2 * sigma_s * sigma_s);
    const float inv_two_sigma_r2 = 1.0f / (2 * sigma_r * sigma_r);
    const int filter_size = 2 * radius + 1;

    // Apply bilateral filter
    for (int y = start_line; y < end_line; y++)
    {
        for (int x = radius; x < width - radius; x++)
        {
            float r_acc = 0.0f;
            float norm_factor = 0.0f;

            float center_value_r = input_r[y * width + x];

            // Iterate through the kernel
            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    int neighbor_x = x + kx;
                    int neighbor_y = y + ky;

                    float neighbor_value_r = input_r[neighbor_y * width + neighbor_x];

                    // Spatial weight
                    float spatial_dist = kx * kx + ky * ky;
                    float spatial_weight = expf(-spatial_dist * inv_two_sigma_s2);

                    // Range weight
                    float range_dist_r = center_value_r - neighbor_value_r;
                    float range_weight_r = expf(-(range_dist_r * range_dist_r) * inv_two_sigma_r2);

                    // Total weight
                    float weight_r = spatial_weight * range_weight_r;

                    // Accumulate
                    r_acc += neighbor_value_r * weight_r;
                    norm_factor += weight_r;
                }
            }

            // Normalize and store the result
            filtered_r[y * width + x] = static_cast<unsigned char>(r_acc / norm_factor);
        }
    }
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Start the MPI
    MPI_Init(&argc, &argv);

    // Get the number of tasks and rank
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    // Read input JPEG image
    const char* input_filepath = argv[1];
    JpegSOA input_jpeg = read_jpeg_soa(input_filepath);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        MPI_Finalize();
        return -1;
    }

    int width = input_jpeg.width;
    int height = input_jpeg.height;

    // Divide the task
    int total_line_num = height - 2;  // Exclude borders
    int line_per_task = total_line_num / numtasks;
    int left_line_num = total_line_num % numtasks;

    std::vector<int> cuts(numtasks + 1, 1);
    int divided_left_line_num = 0;
    for (int i = 0; i < numtasks; i++)
    {
        if (divided_left_line_num < left_line_num)
        {
            cuts[i + 1] = cuts[i] + line_per_task + 1;
            divided_left_line_num++;
        }
        else
            cuts[i + 1] = cuts[i] + line_per_task;
    }

    // Allocate space for filtered image
    unsigned char* filtered_r = new unsigned char[width * height];
    unsigned char* filtered_g = new unsigned char[width * height];
    unsigned char* filtered_b = new unsigned char[width * height];

    // Define filter parameters
    int radius = 1;
    float sigma_s = 15.0f;
    float sigma_r = 30.0f;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform bilateral filtering on each process's part of the image
    bilateral_filter_mpi(filtered_r, input_jpeg.r_values, input_jpeg.g_values, input_jpeg.b_values, 
                         width, height, cuts[taskid], cuts[taskid + 1], radius, sigma_s, sigma_r);
    bilateral_filter_mpi(filtered_g, input_jpeg.g_values, input_jpeg.r_values, input_jpeg.b_values, 
                         width, height, cuts[taskid], cuts[taskid + 1], radius, sigma_s, sigma_r);
    bilateral_filter_mpi(filtered_b, input_jpeg.b_values, input_jpeg.r_values, input_jpeg.g_values, 
                         width, height, cuts[taskid], cuts[taskid + 1], radius, sigma_s, sigma_r);

    // Master process gathers the results from all processes
    if (taskid == MASTER)
    {
        for (int i = 1; i < numtasks; i++)
        {
            MPI_Recv(filtered_r + cuts[i] * width, (cuts[i + 1] - cuts[i]) * width, MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(filtered_g + cuts[i] * width, (cuts[i + 1] - cuts[i]) * width, MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(filtered_b + cuts[i] * width, (cuts[i + 1] - cuts[i]) * width, MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Write the filtered image to file
        const char* output_filepath = argv[2];
        JpegSOA output_jpeg = {filtered_r, filtered_g, filtered_b, width, height, 3, input_jpeg.color_space};
        if (export_jpeg(output_jpeg, output_filepath) != 0)
        {
            std::cerr << "Failed to write output JPEG image\n";
            MPI_Finalize();
            return -1;
        }
    }
    else
    {
        // Send results to the master process
        MPI_Send(filtered_r + cuts[taskid] * width, (cuts[taskid + 1] - cuts[taskid]) * width, MPI_UNSIGNED_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        MPI_Send(filtered_g + cuts[taskid] * width, (cuts[taskid + 1] - cuts[taskid]) * width, MPI_UNSIGNED_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        MPI_Send(filtered_b + cuts[taskid] * width, (cuts[taskid + 1] - cuts[taskid]) * width, MPI_UNSIGNED_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Clean up
    delete[] filtered_r;
    delete[] filtered_g;
    delete[] filtered_b;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    MPI_Finalize();
    return 0;
}