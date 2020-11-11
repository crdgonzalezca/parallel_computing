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
    int width_input, 
    int height_input,
    int channels_input,
    int width_output, 
    int height_output,
    int channels_output) {
    const float x_ratio = (width_input + 0.0) / width_output;
    const float y_ratio = (height_input + 0.0) / height_output;

    //2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    //if ((xIndex < 100) && (yIndex < 100))
    //    printf("%d - %d\n", xIndex, yIndex);

    int px = 0, py = 0; 
    int input_width_step = width_input * channels_input;
    int output_width_step = width_output * channels_output;
    if ((xIndex < width_output) && (yIndex < height_output)){
        py = ceil(yIndex * y_ratio);
        px = ceil(xIndex * x_ratio);
            //ptr_source = img.ptr<uchar>(py);
            
            // Calculate the value of the i,j pixel for each channel
        for (int channel = 0; channel < channels_output; channel++){
            *(output_image + (yIndex * output_width_step + xIndex * channels_output + channel)) =  *(input_image + (py * input_width_step + px * channels_output +  + channel));
        }
    }
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

    // Create result image of 720x480 pixels with 3 channels
    Mat output_image(RESULT_HEIGHT, RESULT_WIDTH, CV_8UC3, Scalar(255, 255, 255)); 
    // Read the image from the given source path
    Mat input_image = imread(source_image_path);
    if(input_image.empty()) {
        printf("Error reading image.");
        return 1;
    }

    // Matrices sizes width * height * 3
    const int input_bytes = input_image.cols * input_image.rows * input_image.channels() * sizeof(unsigned char);
    const int output_bytes = output_image.cols * output_image.rows * output_image.channels() * sizeof(unsigned char);

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

    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_output, output_image.ptr(), output_bytes, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy output image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    
    
    //int threadsPerBlock = 256;
    // int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    int width_input = input_image.cols;
    int height_input = input_image.rows;
    int channels_input = input_image.channels();
    int width_output = output_image.cols;
    int height_output = output_image.rows;
    int channels_output = output_image.channels();

    // Launch the Vector Add CUDA Kernel
    const dim3 threadsPerBlock(16, 16);

	//Calculate numBlocks size to cover the whole image
    const dim3 numBlocks((width_output + threadsPerBlock.x - 1) / threadsPerBlock.x, (height_output + threadsPerBlock.y - 1) / threadsPerBlock.y);
    nearest_neighbour_scaling<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width_input, height_input, channels_input, width_output, height_output, channels_output);
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

