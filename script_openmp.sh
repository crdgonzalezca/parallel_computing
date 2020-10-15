#!/bin/bash

# Compile the code
 g++ image_scaling_openmp.cpp -o image_scaling_openmp -std=c++11 `pkg-config --cflags --libs opencv`  -lpthread -fopenmp -lm

# Run the different experiments
./image_scaling_openmp ./image1_4k.jpg ./image1_nearest_480p.jpg 1 Nearest
./image_scaling_openmp ./image1_4k.jpg ./image1_nearest_480p.jpg 2 Nearest
./image_scaling_openmp ./image1_4k.jpg ./image1_nearest_480p.jpg 4 Nearest
./image_scaling_openmp ./image1_4k.jpg ./image1_nearest_480p.jpg 8 Nearest
./image_scaling_openmp ./image1_4k.jpg ./image1_nearest_480p.jpg 16 Nearest

./image_scaling_openmp ./image2_1080p.jpg ./image2_nearest_480p.jpg 1 Nearest
./image_scaling_openmp ./image2_1080p.jpg ./image2_nearest_480p.jpg 2 Nearest
./image_scaling_openmp ./image2_1080p.jpg ./image2_nearest_480p.jpg 4 Nearest
./image_scaling_openmp ./image2_1080p.jpg ./image2_nearest_480p.jpg 8 Nearest
./image_scaling_openmp ./image2_1080p.jpg ./image2_nearest_480p.jpg 16 Nearest

./image_scaling_openmp ./image3_720p.jpg ./image3_nearest_480p.jpg 1 Nearest
./image_scaling_openmp ./image3_720p.jpg ./image3_nearest_480p.jpg 2 Nearest
./image_scaling_openmp ./image3_720p.jpg ./image3_nearest_480p.jpg 4 Nearest
./image_scaling_openmp ./image3_720p.jpg ./image3_nearest_480p.jpg 8 Nearest
./image_scaling_openmp ./image3_720p.jpg ./image3_nearest_480p.jpg 16 Nearest

./image_scaling_openmp ./image1_4k.jpg ./image1_bilinear_480p.jpg 1 Bilinear
./image_scaling_openmp ./image1_4k.jpg ./image1_bilinear_480p.jpg 2 Bilinear
./image_scaling_openmp ./image1_4k.jpg ./image1_bilinear_480p.jpg 4 Bilinear
./image_scaling_openmp ./image1_4k.jpg ./image1_bilinear_480p.jpg 8 Bilinear
./image_scaling_openmp ./image1_4k.jpg ./image1_bilinear_480p.jpg 16 Bilinear

./image_scaling_openmp ./image2_1080p.jpg ./image2_bilinear_480p.jpg 1 Bilinear
./image_scaling_openmp ./image2_1080p.jpg ./image2_bilinear_480p.jpg 2 Bilinear
./image_scaling_openmp ./image2_1080p.jpg ./image2_bilinear_480p.jpg 4 Bilinear
./image_scaling_openmp ./image2_1080p.jpg ./image2_bilinear_480p.jpg 8 Bilinear
./image_scaling_openmp ./image2_1080p.jpg ./image2_bilinear_480p.jpg 16 Bilinear

./image_scaling_openmp ./image3_720p.jpg ./image3_bilinear_480p.jpg 1 Bilinear
./image_scaling_openmp ./image3_720p.jpg ./image3_bilinear_480p.jpg 2 Bilinear
./image_scaling_openmp ./image3_720p.jpg ./image3_bilinear_480p.jpg 4 Bilinear
./image_scaling_openmp ./image3_720p.jpg ./image3_bilinear_480p.jpg 8 Bilinear
./image_scaling_openmp ./image3_720p.jpg ./image3_bilinear_480p.jpg 16 Bilinear
