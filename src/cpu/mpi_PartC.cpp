#include <memory.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <mpi.h> // MPI Header

#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

// Helper function to clamp pixel values between 0 and 255
inline unsigned char clamp_pixel_value(float value)
{
    if (value < 0)
        return 0;
    if (value > 255)
        return 255;
    return static_cast<unsigned char>(value);
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the number of tasks and my task id
    int numtasks, taskid;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Status status;

    // Read input JPEG File
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
    int num_channels = input_jpeg.num_channels;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Divide the task by rows
    int total_line_num = height - 2;
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

    // Master task
    if (taskid == MASTER)
    {
        std::cout << "Input file from: " << input_filepath << "\n";
        auto filteredImage_r = new unsigned char[width * height];
        auto filteredImage_g = new unsigned char[width * height];
        auto filteredImage_b = new unsigned char[width * height];

        // Master process handles its own section of the image
        for (int y = cuts[taskid]; y < cuts[taskid + 1]; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                // Filter the pixel with bilateral filtering
                float r_sum = 0, g_sum = 0, b_sum = 0;
                float norm_factor = 0;

                for (int ky = -1; ky <= 1; ky++)
                {
                    for (int kx = -1; kx <= 1; kx++)
                    {
                        int neighbor_x = x + kx;
                        int neighbor_y = y + ky;

                        float spatial_weight = expf(-(kx * kx + ky * ky) / (2 * 15.0f * 15.0f));

                        float range_weight_r = expf(-powf(input_jpeg.r_values[y * width + x] - input_jpeg.r_values[neighbor_y * width + neighbor_x], 2) / (2 * 30.0f * 30.0f));
                        float range_weight_g = expf(-powf(input_jpeg.g_values[y * width + x] - input_jpeg.g_values[neighbor_y * width + neighbor_x], 2) / (2 * 30.0f * 30.0f));
                        float range_weight_b = expf(-powf(input_jpeg.b_values[y * width + x] - input_jpeg.b_values[neighbor_y * width + neighbor_x], 2) / (2 * 30.0f * 30.0f));

                        float weight_r = spatial_weight * range_weight_r;
                        float weight_g = spatial_weight * range_weight_g;
                        float weight_b = spatial_weight * range_weight_b;

                        r_sum += input_jpeg.r_values[neighbor_y * width + neighbor_x] * weight_r;
                        g_sum += input_jpeg.g_values[neighbor_y * width + neighbor_x] * weight_g;
                        b_sum += input_jpeg.b_values[neighbor_y * width + neighbor_x] * weight_b;

                        norm_factor += weight_r + weight_g + weight_b;
                    }
                }

                filteredImage_r[y * width + x] = clamp_pixel_value(r_sum / norm_factor);
                filteredImage_g[y * width + x] = clamp_pixel_value(g_sum / norm_factor);
                filteredImage_b[y * width + x] = clamp_pixel_value(b_sum / norm_factor);
            }
        }

                // Receive the filtered image data from other tasks
        for (int i = 1; i < numtasks; i++)
        {
            MPI_Recv(filteredImage_r + cuts[i] * width, (cuts[i + 1] - cuts[i]) * width, MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
            MPI_Recv(filteredImage_g + cuts[i] * width, (cuts[i + 1] - cuts[i]) * width, MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
            MPI_Recv(filteredImage_b + cuts[i] * width, (cuts[i + 1] - cuts[i]) * width, MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        }

        // Save the filtered image
        const char* output_filepath = argv[2];
        JpegSOA output_jpeg{filteredImage_r, filteredImage_g, filteredImage_b, width, height, num_channels};
        export_jpeg(output_jpeg, output_filepath);

        // Cleanup
        delete[] filteredImage_r;
        delete[] filteredImage_g;
        delete[] filteredImage_b;
    }
    // Worker task
    else
    {
        // Each worker processes its own image section
        auto filteredImage_r = new unsigned char[(cuts[taskid + 1] - cuts[taskid]) * width];
        auto filteredImage_g = new unsigned char[(cuts[taskid + 1] - cuts[taskid]) * width];
        auto filteredImage_b = new unsigned char[(cuts[taskid + 1] - cuts[taskid]) * width];

        for (int y = cuts[taskid]; y < cuts[taskid + 1]; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                // Filter the pixel with bilateral filtering
                float r_sum = 0, g_sum = 0, b_sum = 0;
                float norm_factor = 0;

                for (int ky = -1; ky <= 1; ky++)
                {
                    for (int kx = -1; kx <= 1; kx++)
                    {
                        int neighbor_x = x + kx;
                        int neighbor_y = y + ky;

                        float spatial_weight = expf(-(kx * kx + ky * ky) / (2 * 15.0f * 15.0f));

                        float range_weight_r = expf(-powf(input_jpeg.r_values[y * width + x] - input_jpeg.r_values[neighbor_y * width + neighbor_x], 2) / (2 * 30.0f * 30.0f));
                        float range_weight_g = expf(-powf(input_jpeg.g_values[y * width + x] - input_jpeg.g_values[neighbor_y * width + neighbor_x], 2) / (2 * 30.0f * 30.0f));
                        float range_weight_b = expf(-powf(input_jpeg.b_values[y * width + x] - input_jpeg.b_values[neighbor_y * width + neighbor_x], 2) / (2 * 30.0f * 30.0f));

                        float weight_r = spatial_weight * range_weight_r;
                        float weight_g = spatial_weight * range_weight_g;
                        float weight_b = spatial_weight * range_weight_b;

                        r_sum += input_jpeg.r_values[neighbor_y * width + neighbor_x] * weight_r;
                        g_sum += input_jpeg.g_values[neighbor_y * width + neighbor_x] * weight_g;
                        b_sum += input_jpeg.b_values[neighbor_y * width + neighbor_x] * weight_b;

                        norm_factor += weight_r + weight_g + weight_b;
                    }
                }

                filteredImage_r[(y - cuts[taskid]) * width + x] = clamp_pixel_value(r_sum / norm_factor);
                filteredImage_g[(y - cuts[taskid]) * width + x] = clamp_pixel_value(g_sum / norm_factor);
                filteredImage_b[(y - cuts[taskid]) * width + x] = clamp_pixel_value(b_sum / norm_factor);
            }
        }

        // Send the filtered image data back to the master
        MPI_Send(filteredImage_r, (cuts[taskid + 1] - cuts[taskid]) * width, MPI_UNSIGNED_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        MPI_Send(filteredImage_g, (cuts[taskid + 1] - cuts[taskid]) * width, MPI_UNSIGNED_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        MPI_Send(filteredImage_b, (cuts[taskid + 1] - cuts[taskid]) * width, MPI_UNSIGNED_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);

        // Cleanup
        delete[] filteredImage_r;
        delete[] filteredImage_g;
        delete[] filteredImage_b;
    }

    MPI_Finalize();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}