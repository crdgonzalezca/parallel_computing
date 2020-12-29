#!/bin/python3
import subprocess
import os


command = "mpic++ image_scaling_openmpi.cpp -o image_scaling_openmpi -std=c++11 `pkg-config --cflags --libs opencv` -lm"
os.system(command)

_ITERATIONS = 10
time = 0.0
_THREADS = 0
_DIM = 4
images = ["image1_4k.jpg", "image1_1080p.jpg", "image3_720p.jpg"]
algorithms = ["Bilinear", "Nearest"]
with open('results_openmpi.csv', 'w') as f:
    f.write('Time, tasks, image, algorithm\n')
    for algorithm in algorithms:
        for image in images:
            for thread in range(0, 4):
                t = 2**thread
                print(f"Tasks {str(t)}, image: {image}, alg: {algorithm}")
                time = 0.0
                for i in range(_ITERATIONS):
                    command = ["mpirun","--mca", "plm_rsh_no_tree_spawn",  "1", "-np", "4", "--hostfile", "mpi_hosts", "./image_scaling_openmpi", f"./images/{image}", "./images/image1_480p.jpg", algorithm]
                    completed_process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(completed_process.stdout.decode())
                    time += float(completed_process.stdout.decode().split('\n')[-2])
        
                avg = time / _ITERATIONS
                f.write("{:.6f}, {}, {}, {}\n".format(avg, t, image, algorithm))
