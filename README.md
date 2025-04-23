# Distributed CUDA Ray Tracer

This project implements a distributed ray tracing engine using **C++**, **OpenMPI**, and **CUDA**, designed to run in a highly parallelized environment across multiple devices. The goal was to simulate realistic lighting by embracing the computational complexity of ray tracing rather than trying to reduce it.

## Overview

Ray tracing is inherently parallelizable: each pixel's color can be computed independently. This makes it a perfect candidate for distribution across multiple devices and cores. In this implementation:

- **OpenMPI** is used to distribute rendering tasks across devices.
- Each device offloads its portion of work to the **GPU**, leveraging **NVIDIA CUDA** for massive parallelization.
- The project is inspired by [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html), with custom modifications for GPU execution and distributed computation.

## Features

- **3D Primitives**: Supports rendering of spheres and rectangles.
- **Materials**:
  - **Metal** (with adjustable fuzziness)
  - **Glass** (dielectric materials with Snell's Law refraction)
  - **Lambertian** (matte)
  - **Diffuse Light** (emissive surfaces)
- **Antialiasing and Blur**: Achieved via multiple samples per pixel and randomized rays.
- **Recursive Ray Bouncing**: Reflection and refraction are computed recursively for realistic light behavior.
- **Distributed Rendering**:
  - Master-worker architecture using OpenMPI.
  - The image is divided into rows; each worker requests and renders one row at a time.
  - Master collects pixel data and assembles the final image.
- **GPU Acceleration**:
  - CUDA is used to process each row on the GPU.
  - Each pixel's color is computed in parallel using CUDA cores.

## Architecture

### Master-Worker with OpenMPI

- The **master node** (rank 0) keeps track of rows to be rendered and aggregates the final image.
- **Worker nodes** request a row to process, render it using CUDA, and send the results back.
- A “DONE” message signals workers to shut down cleanly once all rows are complete.

### GPU Parallelism with CUDA

- Once a worker receives its row, it dispatches it to the GPU.
- Each pixel in the row is assigned to an individual CUDA core for computation.
- This leverages the full parallel processing capabilities of NVIDIA Jetson devices.

## Dependencies

- C++17
- [OpenMPI](https://www-lb.open-mpi.org/)
- [NVIDIA CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) (tested on Jetson devices)
- CMake (for building the project)

## Executing

### Setting Up Device Cluster

#### Enable Passwordless SSH

OpenMPI requires passwordless SSH between devices in contact with one another. You can achieve this with the following process:

- **Generate an SSH Key Pair** (if not already done) on each device
```bash
ssh-keygen -t rsa -b 4096
```
- Share SSH Keys between devices that will be communicating
```bash
ssh-copy-id <user>@<node-ip>
```
- Ensure you can SSH into the other device without a password
```bash
ssh <user>@<node-ip>
```

#### Create a hosts.txt File

There is an example `hosts.txt` file in this repository. This `hosts.txt` takes advantage of the SSH .config file.

When defining your hosts, you can also define slots, or the number of processes to run. For devices running GPU code, it is advised to run a single process. You can do this by limiting the host to one slot:

```
<hostname> slots=1
```

If you do not define the number of slots, then MPI will automatically use all of the avaliable processes on that device.

### Make the Executable

In the main directory of this project:

```bash
mkdir build && cd build
cmake ..
make
```

### Run Using OpenMPI

Assuming you are running your code from `~` and your code resides in the same directory:

```bash
mpirun --hostfile ./raytracer-cuda/hosts.txt ./raytracer-cuda/build/raytracer
```

If you want to limit the number of processes you are running, you can run the command below. Otherwise, OpenMPI will automatically run all avalible processes defined by the slots in the hostfile.

```bash
mpirun -np <processes> --hostfile ./raytracer-cuda/hosts.txt ./raytracer-cuda/build/raytracer
```

## Performance

The following stats were taken on a simple scene with 6 spheres under varying complexity. The results are given in seconds. For complexity, W is the width of the image, S is the number of samples being taken for each pixel, and D is the recursive ray casting depth. The image has a 16:9 aspect ratio, so the height of the image is dependent on the width.

| Complexity (W, S, D)      | CPU Time (s) | GPU Time (s) | Distributed GPU Time (s)  |
|---------------------------|--------------|--------------|---------------------------|
| W: 400, S: 20, D: 2       | 4.54         | 2.069        | 29.8                      |
| W: 600, S: 50, D: 5       | 25.236       | 45.778       | 343.629                   |
| W: 800, S: 100, D: 10     | 105.68       | 119.78       | 1129.01                   |
| W: 800, S: 100, D: 2      | 90.046       | 69.764       | 558.635                   |
| W: 800, S: 20, D: 20      | 18.313       | 26.517       | 187.859                   |


In the end, my project created a simple ray tracer capable of and successfully running in a highly parallelized environment.

> Christian Wittwer

