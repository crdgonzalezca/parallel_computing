#include "mpi.h"
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

#define RESULT_WIDTH 720
#define RESULT_HEIGHT 480
#define ITERATIONS 10
#define MS 1000000.0
#define MAXTASKS 32

typedef unsigned long long timestamp_t;

// Create result image of 720x480 pixels with 3 channels
Mat result_image(RESULT_HEIGHT, RESULT_WIDTH, CV_8UC3, Scalar(255, 255, 255)); 
Mat img;

/**
Implementation of Nearest Neighbour interpolation algorithm to down 
sample the source image
*/
void nearest_neighbour_scaling(int id, int tasks) {
    const int channels_source = img.channels(), channels_target = result_image.channels(); // Número de canales (3)

    const int width_source = img.size().width;
    const int height_source = img.size().height;
    const int width_target = result_image.size().width;
    const int height_target = result_image.size().height;

    const float x_ratio = (width_source + 0.0) / width_target;
    const float y_ratio = (height_source + 0.0) / height_target;
    int px = 0, py = 0; 
    
    uchar *ptr_source = nullptr;
    uchar *ptr_target = nullptr;

    int n_rows = height_target / tasks;
    int initial_y = n_rows * id;
    int end_y = initial_y + n_rows;

    for (; initial_y < end_y; initial_y++) {
        ptr_target = result_image.ptr<uchar>(initial_y);
        // Iterate over the cols
        for (int j = 0; j < width_target; j++) {
            py = ceil(initial_y * y_ratio);
            px = ceil(j * x_ratio);
            ptr_source = img.ptr<uchar>(py);
            
            // Calculate the value of the i,j pixel for each channel
            for (int channel = 0; channel < channels_target; channel++){
                ptr_target[j * channels_target + channel] =  ptr_source[channels_source * px + channel];
            }
        }
    }
}

/**
Helper function to calculate the pixel in the result image for a specific channel.
*/
int calculate_pixel_value_bilinear(Mat &img, int x, int y, float x_diff, float y_diff, int channel){
    const int channels = img.channels();
    uchar *ptr_img = img.ptr<uchar>(y);
    uchar *ptr_img_2 = img.ptr<uchar>(y + 1);
    int index = channels * x + channel;

    return (
        ptr_img[index] * (1 - x_diff) * (1 - y_diff) +
        ptr_img[index + channels] * x_diff * (1 - y_diff) +
        ptr_img_2[index] * (1 - x_diff) * y_diff + 
        ptr_img_2[index + channels] * x_diff * y_diff
        );
}

/**
Implementation of Bilinear interpolation algorithm to down 
sample the source image.
*/
void bilinear_scaling(int id, int tasks) {
    const int channels_target = result_image.channels();

    const int width_source = img.size().width;
    const int height_source = img.size().height;
    const int width_target = result_image.size().width;
    const int height_target = result_image.size().height;

    const float x_ratio = (width_source + 0.0) / width_target;
    const float y_ratio = (height_source + 0.0) / height_target;
    int px = 0, py = 0;
    float x_diff = 0.0, y_diff = 0.0;
    
    unsigned char *ptr_target = nullptr;

    int n_rows = height_target / tasks;
    int initial_y = n_rows * id;
    int end_y = initial_y + n_rows;

    // Iterate over the rows
    for (; initial_y < end_y; initial_y++) {
        ptr_target = result_image.ptr<uchar>(initial_y);
        // Iterate over the cols
        for (int j = 0; j < width_target; j++) {
            py = (int)(initial_y * y_ratio);
            px = (int)(j * x_ratio);
            x_diff = (x_ratio * j) - px;
            y_diff = (y_ratio * initial_y) - py;

            // Calculate the value of the i,j pixel for each channel
            for (int channel = 0; channel < channels_target; channel++){
                ptr_target[j * channels_target + channel] =  calculate_pixel_value_bilinear(img, px, py, x_diff, y_diff, channel);
            }
        }
    }
}

int main(int argc, char* argv[]) { 
    if (argc != 4) {
        cout << "Arguments are not complete. Usage: image_path image_result_path algorithm" << endl;
        return 1;
    }
    // Read parameters 1- source path, 2- Destination path, 3- algorithm
    string source_image_path = argv[1];
    string result_image_path = argv[2];
    string algorithm = argv[3];

    int tasks, iam, root=0;
    int total_pixels = RESULT_WIDTH * RESULT_HEIGHT * 3;
    double start, end, abs_time, avg_time = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &iam);
    
    int pixels_per_proc = total_pixels / tasks;

    // Read the image from the given source path
    img = imread(source_image_path);
    if(img.empty()) {
        cout << "The image " << source_image_path << " was not found\n";
        return 1;
    }

    start = MPI_Wtime();
    if(algorithm == "Nearest"){
        nearest_neighbour_scaling(iam, tasks);
    }else if(algorithm == "Bilinear"){
        bilinear_scaling(iam, tasks);
    }
    unsigned char *ptr_target = (result_image.ptr() + pixels_per_proc * iam);
    MPI_Gather(ptr_target, pixels_per_proc, MPI_UNSIGNED_CHAR, 
                result_image.ptr(), pixels_per_proc, MPI_UNSIGNED_CHAR, 
                root, MPI_COMM_WORLD);

    end = MPI_Wtime();
    abs_time = end - start;
    MPI_Reduce(&abs_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    
    if (iam == root){
        avg_time /= tasks;
        printf("%f\n", avg_time);
        imwrite(result_image_path, result_image); //Write the image to a file
    }
    MPI_Finalize();
}

// mpic++ image_scaling_openmpi.cpp -o image_scaling_openmpi `pkg-config --cflags --libs opencv` -lm

// mpirun -np 4 ./image_scaling_openmpi ./images/image1_1080p.jpg ./images/image1_480p.jpg Nearest