#include <pthread.h> 
#include <opencv2/opencv.hpp>
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
Mat result_image(RESULT_HEIGHT, RESULT_WIDTH, CV_8UC3, Scalar(255, 255, 255)); 
Mat img;
int THREADS;

timestamp_t get_timestamp (){
    struct timeval now;
    gettimeofday (&now, NULL);
    return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

/**
Implementation of Nearest Neighbour interpolation algorithm to down 
sample the source image
*/
void *nearest_neighbour_scaling(void *idv) {
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

    int id = *(int *)idv;
    int n_raws = height_target / THREADS;
    int initial_y = n_raws * id;
    int end_y = initial_y + n_raws;

    // Iterate over the rows
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
    return 0;
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
void *bilinear_scaling(void *idv) {
    const int channels_target = result_image.channels();

    const int width_source = img.size().width;
    const int height_source = img.size().height;
    const int width_target = result_image.size().width;
    const int height_target = result_image.size().height;

    const float x_ratio = (width_source + 0.0) / width_target;
    const float y_ratio = (height_source + 0.0) / height_target;
    int px = 0, py = 0;
    float x_diff = 0.0, y_diff = 0.0;
    
    uchar *ptr_target = nullptr;

    int id = *(int *)idv;
    int n_raws = height_target / THREADS;
    int initial_y = n_raws * id;
    int end_y = initial_y + n_raws;

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
    return 0;
}

int main(int argc, char* argv[]) { 

    if (argc != 5) {
        cout << "Arguments are not complete. Usage: image_path image_result_path n_threads algorithm" << endl;
        return 1;
    }
    // Read parameters 1- source path, 2- Destination path, 3- Number of threads, 4- algorithm
    string source_image_path = argv[1];
    string result_image_path = argv[2];
    THREADS = atoi(argv[3]);
    string algorithm = argv[4];

    int threadId[THREADS], *retval;    
    pthread_t thread[THREADS];

    vector<timestamp_t> measured_times(ITERATIONS, 0.0);
    timestamp_t start, end;

    // Read the image from the given source path
    img = imread(source_image_path);
    if(img.empty()) {
        return 1;
    }
    
    int r = 0;
    for (int iterator = 0; iterator < ITERATIONS; iterator++){
        start = get_timestamp();
        for(int i = 0; i < THREADS; i++) {  
            threadId[i] = i;
            if(algorithm == "Nearest"){
                r = pthread_create(&thread[i], NULL, nearest_neighbour_scaling, &threadId[i]);
            }else if(algorithm == "Bilinear"){
                r = pthread_create(&thread[i], NULL, bilinear_scaling, &threadId[i]);
            }
            if(r != 0) {
                perror("\n-->No puedo crear hilo");
                exit(-1);
            }
        }

        for(int i = 0; i < THREADS; i++){
            pthread_join(thread[i], (void **)&retval);
        }
        imwrite(result_image_path, result_image); //write the image to a file

        end = get_timestamp();
        measured_times[iterator] = (end - start);
    }

    float avg = 0.0;
    for (timestamp_t &value: measured_times){
        avg += (value + 0.0);
    }
    avg /= ITERATIONS;
    float avg_end = avg / MS;
    printf("Time Elapsed: %f\n", avg_end);

    FILE * fp;
    fp = fopen("results.csv", "a");        
	if (fp==NULL) {fputs ("File error",stderr); exit (1);}
    fprintf(fp,"%f,%d,%s,%s\n",avg_end, THREADS, source_image_path.c_str(), algorithm.c_str());
	fclose ( fp );
    return 0;
}

//   g++ image_scaling.cpp -o image_scaling -std=c++11 `pkg-config --cflags --libs opencv`  -lpthread

//   ./image_scaling ./image1_4k.jpg ./image1_480p.jpg 2 Nearest