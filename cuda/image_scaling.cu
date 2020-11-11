// #include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>
# include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <sys/time.h>

using namespace cv;
using namespace std;

#define RESULT_WIDTH 720
#define RESULT_HEIGHT 480
#define ITERATIONS 10
#define MS 1000000.0

typedef unsigned long long timestamp_t;

// Create result image of 720x480 pixels with 3 channels
Mat output_image(RESULT_HEIGHT, RESULT_WIDTH, CV_8UC3, Scalar(255, 255, 255)); 
Mat input_image;

timestamp_t get_timestamp (){
    struct timeval now;
    gettimeofday (&now, NULL);
    return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(unsigned char *input_image, unsigned char *output_image, 	int width, int height, 
    int inputWidthStep, int outputWidthStep){
    //2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if ((xIndex < width) && (yIndex < height)){
        const int color_tid = yIndex * outputWidthStep + (3 * xIndex);

		//Location of gray pixel in output
		// const int gray_tid = yIndex * outputWidthStep + xIndex;

		const unsigned char blue = input_image[color_tid];
		const unsigned char green = input_image[color_tid + 1];
		const unsigned char red = input_image[color_tid + 2];

		const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;

        output_image[color_tid] = static_cast<unsigned char>(gray);
        output_image[color_tid + 1] = static_cast<unsigned char>(gray);
        output_image[color_tid + 2] = static_cast<unsigned char>(gray);
    }
}

__global__ void nearest_neighbour_scaling(
    unsigned char *input_image, 
    unsigned char *output_image,
    tuple <int, int, int> dims_input, 
    tuple <int, int, int> dims_output,
    int inputWidthStep, 
    int outputWidthStep) {

    const int width_input = dims_input.get(0);
    const int height_input = dims_input.get(1);
    const int channels_input = dims_input.get(2);

    const int width_output = dims_output.get(0);
    const int height_output = dims_output.get(1);
    const int channels_output = dims_output.get(2);

    const float x_ratio = (width_input + 0.0) / width_output;
    const float y_ratio = (height_input + 0.0) / height_output;

    //2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex < 100 && yIndex < 100)
        printf("%d - %d", xIndex, yIndex);

    int px = 0, py = 0; 
    
    //uchar *ptr_source = nullptr;
    //uchar *ptr_target = nullptr;

    //int id = *(int *)idv;
    //int n_raws = height_output / THREADS;
    //int initial_y = n_raws * id;
    //int end_y = initial_y + n_raws;

    // Iterate over the rows
    //for (; initial_y < end_y; initial_y++) {
        //ptr_target = result_image.ptr<uchar>(initial_y);
        // Iterate over the cols
        //for (int j = 0; j < width_output; j++) {
//    if ((xIndex < width) && (yIndex < height)){
//           py = ceil(yIndex * y_ratio);
//            px = ceil(xIndex * x_ratio);
            //ptr_source = img.ptr<uchar>(py);
            
            // Calculate the value of the i,j pixel for each channel
//            for (int channel = 0; channel < channels_output; channel++){
//                ptr_target[j * channels_output + channel] =  ptr_source[channels_input * px + channel];
//            }
//    }
}


/**
 * Host main routine
 */
int main(int argc, char* argv[]) {
    // Read parameters 1- source path, 2- Destination path, 3- Number of threads, 4- algorithm
    if (argc != 4) {
        cout << "Arguments are not complete. Usage: image_path image_result_path n_threads algorithm" << endl;
        return 1;
    }
    string source_image_path = argv[1];
    string result_image_path = argv[2];
    // THREADS = atoi(argv[3]);
    string algorithm = argv[3];

    // time measurement variables
    timestamp_t start, end;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Read the image from the given source path
    input_image = imread(source_image_path);
    if(input_image.empty()) {
        printf("Error reading image.");
        return 1;
    }

    // Matrices sizes width * height * 3
    const int input_bytes = input_image.step * input_image.rows;
    const int output_bytes = output_image.step * output_image.rows;

    unsigned char *d_input, *d_output;
    // Allocate the device input image
//    float *d_A = NULL;
    err = cudaMalloc<unsigned char>(&d_input, input_bytes);
//    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device input image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output image
//    float *d_B = NULL;
    err = cudaMalloc<unsigned char>(&d_output, output_bytes);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device output image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_input, input_image.ptr(), input_bytes, cudaMemcpyHostToDevice);
//    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy input image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    const dim3 block(16, 16);

	//Calculate grid size to cover the whole image
    const dim3 grid((output_image.cols + block.x - 1) / block.x, (output_image.rows + block.y - 1) / block.y);
    
    //int threadsPerBlock = 256;
    // int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    tuple <int, int, int> dims_input = make_tuple(input_image.size().width, input_image.size().height, input_image.channels);
    tuple <int, int, int> dims_output = make_tuple(output_image.size().width, output_image.size().height, output_image.channels);
    nearest_neighbour_scaling<<<grid, block>>>(d_input, d_output, dims_input, dims_output, input_image.step, output_image.step);
//    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(output_image.ptr(), d_output, output_bytes, cudaMemcpyDeviceToHost);
    // err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_input);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    imwrite(result_image_path, output_image); //write the image to a file

    printf("Done\n");
    return 0;
}

