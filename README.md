# Parallel Computing

This repository stores the project for the course Parallel and Distributed Computing at the Universidad Nacional De Colombia. 

This project aims to test different ways to parallelize a algorithm to downsample images in 720p, 1080p and 4K to 480p.

## How to use

First of all, you need to install OpenCV for C++: [How to install](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html)

When dependencies are install, to compile the code:

```bash
g++ image_scaling.cpp -o image_scaling -std=c++11 `pkg-config --cflags --libs opencv`
```

To run:

```bash
./image_scaling ./image1_720p.jpg ./result.jpg 4
```

This accepts 3 parameters:

1. Source input image path.
2. Result image path.
3. Number of threads.