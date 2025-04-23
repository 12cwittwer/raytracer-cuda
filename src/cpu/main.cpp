#include <iostream>
#include <mpi.h>
#include "raytracer/bvh.h"
#include "raytracer/bvh_builder.h"
#include "raytracer/camera.h"
#include "raytracer/hittable.h"
#include "raytracer/hittable_list.h"
#include "raytracer/material.h"
#include "raytracer/quad.h"
#include "raytracer/sphere_gpu.h"
#include "raytracer/instances.h"

#include "raytracer/cuda_utils.h"  // for CUDA_CHECK

int WIDTH = 800;
int SAMPLES = 50;
int DEPTH = 5;

void mpi() {
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // === Camera Setup ===
    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = WIDTH;
    cam.samples_per_pixel = SAMPLES;
    cam.max_depth = DEPTH;
    cam.background = color(1.0, 1.0, 1.0);
    cam.vfov = 20;
    cam.lookfrom = point3(0, 0, 0);
    cam.lookat = point3(0, 0, -1);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 0;

    if (rank == 0) {
        // MASTER
        cam.delegate(rank, num_procs);
        MPI_Finalize();
        return;
    }


    const int max_glass = 2;
    const int max_metals = 2;
    const int max_lambertians = 2;
    const int max_spheres = 6;
    const int max_materials = 6;
    const int max_objects = 6;
    const int max_hittable_list = 1;

    dielectric* glasses = new dielectric[max_glass];
    metal* metals = new metal[max_metals];
    lambertian* lambertians = new lambertian[max_lambertians];
    material* materials = new material[max_materials];
    gpu_sphere* spheres = new gpu_sphere[max_spheres];
    hittable* objects = new hittable[max_objects];
    gpu_hittable_list* hittable_lists = new gpu_hittable_list[max_hittable_list];

    int glass_count = 0;
    int metal_count = 0;
    int lambertian_count = 0;
    int material_count = 0;
    int sphere_count = 0;
    int object_count = 0;
    int hittable_list_count = 0;

    // === Build Scene ===
    glasses[glass_count++] = dielectric{1.5};
    glasses[glass_count++] = dielectric{1.3};

    metals[metal_count++] = metal{color(0.8, 0.3, 0.3), 0.05};
    metals[metal_count++] = metal{color(0.7, 0.1, 0.7), 0.05};

    lambertians[lambertian_count++] = lambertian{color(0.1, 0.1, 0.8)};
    lambertians[lambertian_count++] = lambertian{color(0.0, 0.6, 0.0)};

    materials[material_count++] = material{material_type::dielectric, nullptr};
    materials[material_count++] = material{material_type::dielectric, nullptr};
    materials[material_count++] = material{material_type::metal, nullptr};
    materials[material_count++] = material{material_type::metal, nullptr};
    materials[material_count++] = material{material_type::lambertian, nullptr};
    materials[material_count++] = material{material_type::lambertian, nullptr};

    spheres[sphere_count++] = gpu_sphere(point3(-1.0, 0, -9), 0.5, nullptr); // glass 1
    spheres[sphere_count++] = gpu_sphere(point3(0.0, 0, -8), 0.2, nullptr); // glass 2
    spheres[sphere_count++] = gpu_sphere(point3(0.0, 0, -10), 0.5, nullptr);  // metal 1
    spheres[sphere_count++] = gpu_sphere(point3(1.0, 0, -9), 0.5, nullptr);  // metal 2
    spheres[sphere_count++] = gpu_sphere(point3(0, 1.0, -9), 0.5, nullptr);  // lambertian 1
    spheres[sphere_count++] = gpu_sphere(point3(0, -900, -15), 899.5, nullptr);  // lambertian 2 (floor)

    for (int i = 0; i < max_objects; i++) {
        objects[object_count++] = hittable{hittable_type::sphere, nullptr};
    }

    hittable_lists[hittable_list_count++] = gpu_hittable_list{nullptr, object_count};

    // === Allocate Device Memory ===
    dielectric* d_glasses;
    metal* d_metals;
    lambertian* d_lambertians;
    material* d_materials;
    gpu_sphere* d_spheres;
    hittable* d_objects;
    gpu_hittable_list* d_hittable_lists;

    CUDA_CHECK(cudaMalloc(&d_glasses, glass_count * sizeof(dielectric)));
    CUDA_CHECK(cudaMalloc(&d_metals, metal_count * sizeof(metal)));
    CUDA_CHECK(cudaMalloc(&d_lambertians, lambertian_count * sizeof(lambertian)));
    CUDA_CHECK(cudaMalloc(&d_materials, material_count * sizeof(material)));
    CUDA_CHECK(cudaMalloc(&d_spheres, sphere_count * sizeof(gpu_sphere)));
    CUDA_CHECK(cudaMalloc(&d_objects, object_count * sizeof(hittable)));
    CUDA_CHECK(cudaMalloc(&d_hittable_lists, hittable_list_count * sizeof(gpu_hittable_list)));

    // === Copy Materials ===
    CUDA_CHECK(cudaMemcpy(d_glasses, glasses, glass_count * sizeof(dielectric), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metals, metals, metal_count * sizeof(metal), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lambertians, lambertians, lambertian_count * sizeof(lambertian), cudaMemcpyHostToDevice));

    materials[0].data = reinterpret_cast<void*>(d_glasses + 0);
    materials[1].data = reinterpret_cast<void*>(d_glasses + 1);
    materials[2].data = reinterpret_cast<void*>(d_metals + 0);
    materials[3].data = reinterpret_cast<void*>(d_metals + 1);
    materials[4].data = reinterpret_cast<void*>(d_lambertians + 0);
    materials[5].data = reinterpret_cast<void*>(d_lambertians + 1);
    

    CUDA_CHECK(cudaMemcpy(d_materials, materials, material_count * sizeof(material), cudaMemcpyHostToDevice));

    // === Copy Spheres with material pointers ===
    spheres[0].mat_ptr = d_materials + 0;
    spheres[1].mat_ptr = d_materials + 1;
    spheres[2].mat_ptr = d_materials + 2;
    spheres[3].mat_ptr = d_materials + 3;
    spheres[4].mat_ptr = d_materials + 4;
    spheres[5].mat_ptr = d_materials + 5;

    CUDA_CHECK(cudaMemcpy(d_spheres, spheres, sphere_count * sizeof(gpu_sphere), cudaMemcpyHostToDevice));

    for (int i = 0; i < object_count; i++) {
        objects[i].data = reinterpret_cast<void*>(d_spheres + i);
    }

    CUDA_CHECK(cudaMemcpy(d_objects, objects, object_count * sizeof(hittable), cudaMemcpyHostToDevice));

    hittable_lists[0].objects = d_objects;

    CUDA_CHECK(cudaMemcpy(d_hittable_lists, hittable_lists, hittable_list_count * sizeof(gpu_hittable_list), cudaMemcpyHostToDevice));

    // === World ===
    hittable world = hittable{
        hittable_type::hittable_list,
        reinterpret_cast<void*>(d_hittable_lists)
    };

    hittable* d_world;
    CUDA_CHECK(cudaMalloc(&d_world, sizeof(hittable)));
    CUDA_CHECK(cudaMemcpy(d_world, &world, sizeof(hittable), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 32768));

    cam.render_gpu(d_world, rank, num_procs);

    // === Cleanup Device ===
    cudaFree(d_glasses);
    cudaFree(d_metals);
    cudaFree(d_lambertians);
    cudaFree(d_materials);
    cudaFree(d_spheres);
    cudaFree(d_objects);
    cudaFree(d_hittable_lists);
    cudaFree(d_world);

    // === Cleanup Host ===
    delete[] glasses;
    delete[] metals;
    delete[] lambertians;
    delete[] materials;
    delete[] spheres;
    delete[] objects;
    delete[] hittable_lists;

    MPI_Finalize();
}

void cpu() {
    // === Camera Setup ===
    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = WIDTH;
    cam.samples_per_pixel = SAMPLES;
    cam.max_depth = DEPTH;
    cam.background = color(1.0, 1.0, 1.0);
    cam.vfov = 20;
    cam.lookfrom = point3(0, 0, 0);
    cam.lookat = point3(0, 0, -1);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 0;

    // === World Structure ===

    const int max_glass = 2;
    const int max_metals = 2;
    const int max_lambertians = 2;
    const int max_spheres = 6;
    const int max_materials = 6;
    const int max_objects = 6;
    const int max_hittable_list = 1;

    dielectric* glasses = new dielectric[max_glass];
    metal* metals = new metal[max_metals];
    lambertian* lambertians = new lambertian[max_lambertians];
    material* materials = new material[max_materials];
    gpu_sphere* spheres = new gpu_sphere[max_spheres];
    hittable* objects = new hittable[max_objects];
    gpu_hittable_list* hittable_lists = new gpu_hittable_list[max_hittable_list];

    int glass_count = 0;
    int metal_count = 0;
    int lambertian_count = 0;
    int material_count = 0;
    int sphere_count = 0;
    int object_count = 0;
    int hittable_list_count = 0;

    // === Build Scene ===
    glasses[glass_count++] = dielectric{1.5};
    glasses[glass_count++] = dielectric{1.3};

    metals[metal_count++] = metal{color(0.8, 0.3, 0.3), 0.05};
    metals[metal_count++] = metal{color(0.7, 0.1, 0.7), 0.05};

    lambertians[lambertian_count++] = lambertian{color(0.1, 0.1, 0.8)};
    lambertians[lambertian_count++] = lambertian{color(0.0, 0.6, 0.0)};

    materials[material_count++] = material{material_type::dielectric, (void*)&glasses[0]};
    materials[material_count++] = material{material_type::dielectric, (void*)&glasses[1]};
    materials[material_count++] = material{material_type::metal, (void*)&metals[0]};
    materials[material_count++] = material{material_type::metal, (void*)&metals[1]};
    materials[material_count++] = material{material_type::lambertian, (void*)&lambertians[0]};
    materials[material_count++] = material{material_type::lambertian, (void*)&lambertians[1]};

    spheres[sphere_count++] = gpu_sphere(point3(-1.0, 0, -9), 0.5, &materials[0]);
    spheres[sphere_count++] = gpu_sphere(point3(0.0, 0, -8), 0.2, &materials[1]);
    spheres[sphere_count++] = gpu_sphere(point3(0.0, 0, -10), 0.5, &materials[2]);
    spheres[sphere_count++] = gpu_sphere(point3(1.0, 0, -9), 0.5, &materials[3]);
    spheres[sphere_count++] = gpu_sphere(point3(0, 1.0, -9), 0.5, &materials[4]);
    spheres[sphere_count++] = gpu_sphere(point3(0, -900, -15), 899.5, &materials[5]);

    for (int i = 0; i < max_objects; i++) {
        objects[object_count++] = hittable{hittable_type::sphere, (void*)&spheres[i]};
    }

    hittable_lists[hittable_list_count++] = gpu_hittable_list{objects, object_count};

    hittable world = hittable{hittable_type::hittable_list, (void*)&hittable_lists[0]};

    cam.render(&world);

    // === Cleanup Host ===
    delete[] glasses;
    delete[] metals;
    delete[] lambertians;
    delete[] materials;
    delete[] spheres;
    delete[] objects;
    delete[] hittable_lists;
}

void whole_image() {;

    // === Camera Setup ===
    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = WIDTH;
    cam.samples_per_pixel = SAMPLES;
    cam.max_depth = DEPTH;
    cam.background = color(1.0, 1.0, 1.0);
    cam.vfov = 20;
    cam.lookfrom = point3(0, 0, 0);
    cam.lookat = point3(0, 0, -1);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 0;


    const int max_glass = 2;
    const int max_metals = 2;
    const int max_lambertians = 2;
    const int max_spheres = 6;
    const int max_materials = 6;
    const int max_objects = 6;
    const int max_hittable_list = 1;

    dielectric* glasses = new dielectric[max_glass];
    metal* metals = new metal[max_metals];
    lambertian* lambertians = new lambertian[max_lambertians];
    material* materials = new material[max_materials];
    gpu_sphere* spheres = new gpu_sphere[max_spheres];
    hittable* objects = new hittable[max_objects];
    gpu_hittable_list* hittable_lists = new gpu_hittable_list[max_hittable_list];

    int glass_count = 0;
    int metal_count = 0;
    int lambertian_count = 0;
    int material_count = 0;
    int sphere_count = 0;
    int object_count = 0;
    int hittable_list_count = 0;

    // === Build Scene ===
    glasses[glass_count++] = dielectric{1.5};
    glasses[glass_count++] = dielectric{1.3};

    metals[metal_count++] = metal{color(0.8, 0.3, 0.3), 0.05};
    metals[metal_count++] = metal{color(0.7, 0.1, 0.7), 0.05};

    lambertians[lambertian_count++] = lambertian{color(0.1, 0.1, 0.8)};
    lambertians[lambertian_count++] = lambertian{color(0.0, 0.6, 0.0)};

    materials[material_count++] = material{material_type::dielectric, nullptr};
    materials[material_count++] = material{material_type::dielectric, nullptr};
    materials[material_count++] = material{material_type::metal, nullptr};
    materials[material_count++] = material{material_type::metal, nullptr};
    materials[material_count++] = material{material_type::lambertian, nullptr};
    materials[material_count++] = material{material_type::lambertian, nullptr};

    spheres[sphere_count++] = gpu_sphere(point3(-1.0, 0, -9), 0.5, nullptr); // glass 1
    spheres[sphere_count++] = gpu_sphere(point3(0.0, 0, -8), 0.2, nullptr); // glass 2
    spheres[sphere_count++] = gpu_sphere(point3(0.0, 0, -10), 0.5, nullptr);  // metal 1
    spheres[sphere_count++] = gpu_sphere(point3(1.0, 0, -9), 0.5, nullptr);  // metal 2
    spheres[sphere_count++] = gpu_sphere(point3(0, 1.0, -9), 0.5, nullptr);  // lambertian 1
    spheres[sphere_count++] = gpu_sphere(point3(0, -900, -15), 899.5, nullptr);  // lambertian 2 (floor)

    for (int i = 0; i < max_objects; i++) {
        objects[object_count++] = hittable{hittable_type::sphere, nullptr};
    }

    hittable_lists[hittable_list_count++] = gpu_hittable_list{nullptr, object_count};

    // === Allocate Device Memory ===
    dielectric* d_glasses;
    metal* d_metals;
    lambertian* d_lambertians;
    material* d_materials;
    gpu_sphere* d_spheres;
    hittable* d_objects;
    gpu_hittable_list* d_hittable_lists;

    CUDA_CHECK(cudaMalloc(&d_glasses, glass_count * sizeof(dielectric)));
    CUDA_CHECK(cudaMalloc(&d_metals, metal_count * sizeof(metal)));
    CUDA_CHECK(cudaMalloc(&d_lambertians, lambertian_count * sizeof(lambertian)));
    CUDA_CHECK(cudaMalloc(&d_materials, material_count * sizeof(material)));
    CUDA_CHECK(cudaMalloc(&d_spheres, sphere_count * sizeof(gpu_sphere)));
    CUDA_CHECK(cudaMalloc(&d_objects, object_count * sizeof(hittable)));
    CUDA_CHECK(cudaMalloc(&d_hittable_lists, hittable_list_count * sizeof(gpu_hittable_list)));

    // === Copy Materials ===
    CUDA_CHECK(cudaMemcpy(d_glasses, glasses, glass_count * sizeof(dielectric), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metals, metals, metal_count * sizeof(metal), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lambertians, lambertians, lambertian_count * sizeof(lambertian), cudaMemcpyHostToDevice));

    materials[0].data = reinterpret_cast<void*>(d_glasses + 0);
    materials[1].data = reinterpret_cast<void*>(d_glasses + 1);
    materials[2].data = reinterpret_cast<void*>(d_metals + 0);
    materials[3].data = reinterpret_cast<void*>(d_metals + 1);
    materials[4].data = reinterpret_cast<void*>(d_lambertians + 0);
    materials[5].data = reinterpret_cast<void*>(d_lambertians + 1);
    

    CUDA_CHECK(cudaMemcpy(d_materials, materials, material_count * sizeof(material), cudaMemcpyHostToDevice));

    // === Copy Spheres with material pointers ===
    spheres[0].mat_ptr = d_materials + 0;
    spheres[1].mat_ptr = d_materials + 1;
    spheres[2].mat_ptr = d_materials + 2;
    spheres[3].mat_ptr = d_materials + 3;
    spheres[4].mat_ptr = d_materials + 4;
    spheres[5].mat_ptr = d_materials + 5;

    CUDA_CHECK(cudaMemcpy(d_spheres, spheres, sphere_count * sizeof(gpu_sphere), cudaMemcpyHostToDevice));

    for (int i = 0; i < object_count; i++) {
        objects[i].data = reinterpret_cast<void*>(d_spheres + i);
    }

    CUDA_CHECK(cudaMemcpy(d_objects, objects, object_count * sizeof(hittable), cudaMemcpyHostToDevice));

    hittable_lists[0].objects = d_objects;

    CUDA_CHECK(cudaMemcpy(d_hittable_lists, hittable_lists, hittable_list_count * sizeof(gpu_hittable_list), cudaMemcpyHostToDevice));

    // === World ===
    hittable world = hittable{
        hittable_type::hittable_list,
        reinterpret_cast<void*>(d_hittable_lists)
    };

    hittable* d_world;
    CUDA_CHECK(cudaMalloc(&d_world, sizeof(hittable)));
    CUDA_CHECK(cudaMemcpy(d_world, &world, sizeof(hittable), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 32768));

    cam.render_whole(d_world);

    // === Cleanup Device ===
    cudaFree(d_glasses);
    cudaFree(d_metals);
    cudaFree(d_lambertians);
    cudaFree(d_materials);
    cudaFree(d_spheres);
    cudaFree(d_objects);
    cudaFree(d_hittable_lists);
    cudaFree(d_world);

    // === Cleanup Host ===
    delete[] glasses;
    delete[] metals;
    delete[] lambertians;
    delete[] materials;
    delete[] spheres;
    delete[] objects;
    delete[] hittable_lists;
}

<<<<<<< HEAD
void mpi() {

    // === Camera Setup ===
    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 800;
    cam.samples_per_pixel = 100;
    cam.max_depth = 10;
    cam.background = color(1.0, 1.0, 1.0);
=======
void light() {
    // === Camera Setup ===
    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = WIDTH;
    cam.samples_per_pixel = SAMPLES;
    cam.max_depth = DEPTH;
    cam.background = color(0.0, 0.0, 0.0);
>>>>>>> mpi
    cam.vfov = 20;
    cam.lookfrom = point3(0, 0, 0);
    cam.lookat = point3(0, 0, -1);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 0;

<<<<<<< HEAD
    const int max_glass = 2;
    const int max_metals = 2;
    const int max_lambertians = 2;
    const int max_spheres = 6;
    const int max_materials = 6;
    const int max_objects = 6;
=======
    // === World Structure ===
    const int max_glass = 2;
    const int max_metals = 2;
    const int max_lambertians = 2;
    const int max_light = 2;
    const int max_quad = 2;
    const int max_spheres = 6;
    const int max_materials = 8;
    const int max_objects = 8;
>>>>>>> mpi
    const int max_hittable_list = 1;

    dielectric* glasses = new dielectric[max_glass];
    metal* metals = new metal[max_metals];
    lambertian* lambertians = new lambertian[max_lambertians];
<<<<<<< HEAD
    material* materials = new material[max_materials];
=======
    diffuse_light* lights = new diffuse_light[max_light];
    material* materials = new material[max_materials];
    quad* quads = new quad[max_quad];
>>>>>>> mpi
    gpu_sphere* spheres = new gpu_sphere[max_spheres];
    hittable* objects = new hittable[max_objects];
    gpu_hittable_list* hittable_lists = new gpu_hittable_list[max_hittable_list];

    int glass_count = 0;
    int metal_count = 0;
<<<<<<< HEAD
    int lambertian_count = 0;
    int material_count = 0;
    int sphere_count = 0;
=======
    int light_count = 0;
    int lambertian_count = 0;
    int material_count = 0;
    int sphere_count = 0;
    int quad_count = 0;
>>>>>>> mpi
    int object_count = 0;
    int hittable_list_count = 0;

    // === Build Scene ===
    glasses[glass_count++] = dielectric{1.5};
    glasses[glass_count++] = dielectric{1.3};

<<<<<<< HEAD
    metals[metal_count++] = metal{color(0.8, 0.3, 0.3), 0.05};
=======
    metals[metal_count++] = metal{color(1, 1, 1), 0.00};
>>>>>>> mpi
    metals[metal_count++] = metal{color(0.7, 0.1, 0.7), 0.05};

    lambertians[lambertian_count++] = lambertian{color(0.1, 0.1, 0.8)};
    lambertians[lambertian_count++] = lambertian{color(0.0, 0.6, 0.0)};

<<<<<<< HEAD
    materials[material_count++] = material{material_type::dielectric, nullptr};
    materials[material_count++] = material{material_type::dielectric, nullptr};
    materials[material_count++] = material{material_type::metal, nullptr};
    materials[material_count++] = material{material_type::metal, nullptr};
    materials[material_count++] = material{material_type::lambertian, nullptr};
    materials[material_count++] = material{material_type::lambertian, nullptr};

    spheres[sphere_count++] = gpu_sphere(point3(-1.0, 0, -9), 0.5, nullptr); // glass 1
    spheres[sphere_count++] = gpu_sphere(point3(0.0, 0, -8), 0.2, nullptr); // glass 2
    spheres[sphere_count++] = gpu_sphere(point3(0.0, 0, -10), 0.5, nullptr);  // metal 1
    spheres[sphere_count++] = gpu_sphere(point3(1.0, 0, -9), 0.5, nullptr);  // metal 2
    spheres[sphere_count++] = gpu_sphere(point3(0, 1.0, -9), 0.5, nullptr);  // lambertian 1
    spheres[sphere_count++] = gpu_sphere(point3(0, -900, -15), 899.5, nullptr);  // lambertian 2 (floor)

    for (int i = 0; i < max_objects; i++) {
        objects[object_count++] = hittable{hittable_type::sphere, nullptr};
    }

    hittable_lists[hittable_list_count++] = gpu_hittable_list{nullptr, object_count};

    // === Allocate Device Memory ===
    dielectric* d_glasses;
    metal* d_metals;
    lambertian* d_lambertians;
    material* d_materials;
    gpu_sphere* d_spheres;
    hittable* d_objects;
    gpu_hittable_list* d_hittable_lists;

    CUDA_CHECK(cudaMalloc(&d_glasses, glass_count * sizeof(dielectric)));
    CUDA_CHECK(cudaMalloc(&d_metals, metal_count * sizeof(metal)));
    CUDA_CHECK(cudaMalloc(&d_lambertians, lambertian_count * sizeof(lambertian)));
    CUDA_CHECK(cudaMalloc(&d_materials, material_count * sizeof(material)));
    CUDA_CHECK(cudaMalloc(&d_spheres, sphere_count * sizeof(gpu_sphere)));
    CUDA_CHECK(cudaMalloc(&d_objects, object_count * sizeof(hittable)));
    CUDA_CHECK(cudaMalloc(&d_hittable_lists, hittable_list_count * sizeof(gpu_hittable_list)));

    // === Copy Materials ===
    CUDA_CHECK(cudaMemcpy(d_glasses, glasses, glass_count * sizeof(dielectric), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metals, metals, metal_count * sizeof(metal), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lambertians, lambertians, lambertian_count * sizeof(lambertian), cudaMemcpyHostToDevice));

    materials[0].data = reinterpret_cast<void*>(d_glasses + 0);
    materials[1].data = reinterpret_cast<void*>(d_glasses + 1);
    materials[2].data = reinterpret_cast<void*>(d_metals + 0);
    materials[3].data = reinterpret_cast<void*>(d_metals + 1);
    materials[4].data = reinterpret_cast<void*>(d_lambertians + 0);
    materials[5].data = reinterpret_cast<void*>(d_lambertians + 1);
    

    CUDA_CHECK(cudaMemcpy(d_materials, materials, material_count * sizeof(material), cudaMemcpyHostToDevice));

    // === Copy Spheres with material pointers ===
    spheres[0].mat_ptr = d_materials + 0;
    spheres[1].mat_ptr = d_materials + 1;
    spheres[2].mat_ptr = d_materials + 2;
    spheres[3].mat_ptr = d_materials + 3;
    spheres[4].mat_ptr = d_materials + 4;
    spheres[5].mat_ptr = d_materials + 5;

    CUDA_CHECK(cudaMemcpy(d_spheres, spheres, sphere_count * sizeof(gpu_sphere), cudaMemcpyHostToDevice));

    for (int i = 0; i < object_count; i++) {
        objects[i].data = reinterpret_cast<void*>(d_spheres + i);
    }

    CUDA_CHECK(cudaMemcpy(d_objects, objects, object_count * sizeof(hittable), cudaMemcpyHostToDevice));

    hittable_lists[0].objects = d_objects;

    CUDA_CHECK(cudaMemcpy(d_hittable_lists, hittable_lists, hittable_list_count * sizeof(gpu_hittable_list), cudaMemcpyHostToDevice));

    // === World ===
    hittable world = hittable{
        hittable_type::hittable_list,
        reinterpret_cast<void*>(d_hittable_lists)
    };

    hittable* d_world;
    CUDA_CHECK(cudaMalloc(&d_world, sizeof(hittable)));
    CUDA_CHECK(cudaMemcpy(d_world, &world, sizeof(hittable), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 500000));

    cam.render_gpu(d_world);

    // === Cleanup Device ===
    cudaFree(d_glasses);
    cudaFree(d_metals);
    cudaFree(d_lambertians);
    cudaFree(d_materials);
    cudaFree(d_spheres);
    cudaFree(d_objects);
    cudaFree(d_hittable_lists);
    cudaFree(d_world);
=======
    lights[light_count++] = diffuse_light{color(1.0, 1.0, 1.0)};
    lights[light_count++] = diffuse_light{color(0.8, 0.1, 0.3)};

    materials[material_count++] = material{material_type::dielectric, (void*)&glasses[0]};
    materials[material_count++] = material{material_type::dielectric, (void*)&glasses[1]};
    materials[material_count++] = material{material_type::metal, (void*)&metals[0]};
    materials[material_count++] = material{material_type::metal, (void*)&metals[1]};
    materials[material_count++] = material{material_type::lambertian, (void*)&lambertians[0]};
    materials[material_count++] = material{material_type::lambertian, (void*)&lambertians[1]};
    materials[material_count++] = material{material_type::diffuse_light, (void*)&lights[0]};
    materials[material_count++] = material{material_type::diffuse_light, (void*)&lights[1]};

    // Spheres
    spheres[sphere_count++] = gpu_sphere(point3(1.0, 0, -9), 0.5, &materials[0]);
    spheres[sphere_count++] = gpu_sphere(point3(-100.0, 0, -8), 0.2, &materials[1]);
    spheres[sphere_count++] = gpu_sphere(point3(-100.0, 0, -10), 0.5, &materials[2]);
    spheres[sphere_count++] = gpu_sphere(point3(1, 0, -9), 0.5, &materials[3]);
    spheres[sphere_count++] = gpu_sphere(point3(-100, 1.0, -9), 0.5, &materials[4]);
    spheres[sphere_count++] = gpu_sphere(point3(0, -900, -15), 899.5, &materials[2]);

    for (int i = 0; i < sphere_count && object_count < max_objects; i++) {
        objects[object_count++] = hittable{hittable_type::sphere, (void*)&spheres[i]};
    }

    // Quads (light panels in view)
    quads[quad_count++] = quad(point3(-1, 1.5, -11), vec3(2, 0, 0), vec3(0, -1, -1), &materials[6]);
    quads[quad_count++] = quad(point3(-100.0, -1.0, -7.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), &materials[7]);

    for (int i = 0; i < quad_count && object_count < max_objects; i++) {
        objects[object_count++] = hittable{hittable_type::quad, (void*)&quads[i]};
    }

    hittable_lists[hittable_list_count++] = gpu_hittable_list{objects, object_count};

    hittable world = hittable{hittable_type::hittable_list, (void*)&hittable_lists[0]};

    cam.render(&world);
>>>>>>> mpi

    // === Cleanup Host ===
    delete[] glasses;
    delete[] metals;
    delete[] lambertians;
<<<<<<< HEAD
=======
    delete[] lights;
>>>>>>> mpi
    delete[] materials;
    delete[] spheres;
    delete[] objects;
    delete[] hittable_lists;
}

<<<<<<< HEAD



int main() {
    switch (15) {
        case 1: spheres();      break;
        case 2: quads();        break;
        case 3: light();        break;
        case 4: cornell();      break;
        case 5: final();        break;
        case 6: glass_box();    break;
        case 7: glass_orb();    break;
        case 8: final_scene();  break; // Seg Fault
        case 9: my_scene();     break;
        case 10: small_boi();   break;
        case 11: sboi();        break;
        case 12: s_metal();     break;
        case 13: two_spheres(); break;
        case 14: four_spheres();break;
        case 15: final_sphere();break;
        case 16: mpi();         break;
=======
void floating() {
    // === Camera Setup ===
    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = WIDTH;
    cam.samples_per_pixel = SAMPLES;
    cam.max_depth = DEPTH;
    cam.background = color(0.0, 0.0, 0.0);
    cam.vfov = 25;
    cam.lookfrom = point3(0, 1, 3);
    cam.lookat = point3(0, 0.5, -1);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 0;

    // === World Structure ===
    const int max_glass = 1;
    const int max_metals = 2;
    const int max_lambertians = 1;
    const int max_light = 3;
    const int max_quad = 3;
    const int max_spheres = 3;
    const int max_materials = 7;
    const int max_objects = 8;
    const int max_hittable_list = 1;

    dielectric* glasses = new dielectric[max_glass];
    metal* metals = new metal[max_metals];
    lambertian* lambertians = new lambertian[max_lambertians];
    diffuse_light* lights = new diffuse_light[max_light];
    material* materials = new material[max_materials];
    quad* quads = new quad[max_quad];
    gpu_sphere* spheres = new gpu_sphere[max_spheres];
    hittable* objects = new hittable[max_objects];
    gpu_hittable_list* hittable_lists = new gpu_hittable_list[max_hittable_list];

    int glass_count = 0;
    int metal_count = 0;
    int light_count = 0;
    int lambertian_count = 0;
    int material_count = 0;
    int sphere_count = 0;
    int quad_count = 0;
    int object_count = 0;
    int hittable_list_count = 0;

    // === Materials ===
    glasses[glass_count++] = dielectric{1.5};
    metals[metal_count++] = metal{color(0.95, 0.95, 0.95), 0.01};  // Mirror
    metals[metal_count++] = metal{color(0.5, 0.5, 0.5), 0.3};      // Rough metal
    lambertians[lambertian_count++] = lambertian{color(0.2, 0.2, 0.8)}; // Decorative

    lights[light_count++] = diffuse_light{color(6.0, 6.0, 6.0)}; // Bright white
    lights[light_count++] = diffuse_light{color(0.8, 0.1, 0.1)}; // Warm
    lights[light_count++] = diffuse_light{color(0.1, 0.1, 1.8)}; // Cool blue

    materials[material_count++] = material{material_type::dielectric, (void*)&glasses[0]};
    materials[material_count++] = material{material_type::metal, (void*)&metals[0]};
    materials[material_count++] = material{material_type::metal, (void*)&metals[1]};
    materials[material_count++] = material{material_type::lambertian, (void*)&lambertians[0]};
    materials[material_count++] = material{material_type::diffuse_light, (void*)&lights[0]};
    materials[material_count++] = material{material_type::diffuse_light, (void*)&lights[1]};
    materials[material_count++] = material{material_type::diffuse_light, (void*)&lights[2]};

    // === Geometry ===

    // Floating crystal sphere
    spheres[sphere_count++] = gpu_sphere(point3(0, 0, -5), 1, &materials[0]);

    // Mirror floor
    spheres[sphere_count++] = gpu_sphere(point3(0, -1000, -5), 999.0, &materials[1]);

    // Accent orb
    spheres[sphere_count++] = gpu_sphere(point3(1.2, 0.4, -4.5), 0.3, &materials[2]);

    for (int i = 0; i < sphere_count && object_count < max_objects; i++) {
        objects[object_count++] = hittable{hittable_type::sphere, (void*)&spheres[i]};
    }

    // Light panels (walls and ceiling)
    quads[quad_count++] = quad(point3(-1.5, 1.5, -5.0), vec3(3, 0, 0), vec3(0, 0, -1), &materials[4]); // top
    quads[quad_count++] = quad(point3(-2.0, 0, -6.0), vec3(0, 1, 0), vec3(0, 0, 3), &materials[5]);   // left warm
    quads[quad_count++] = quad(point3(2.0, 0, -6.0), vec3(0, 1, 0), vec3(0, 0, 3), &materials[6]);    // right cool

    for (int i = 0; i < quad_count && object_count < max_objects; i++) {
        objects[object_count++] = hittable{hittable_type::quad, (void*)&quads[i]};
    }

    // === World and Render ===
    hittable_lists[hittable_list_count++] = gpu_hittable_list{objects, object_count};
    hittable world = hittable{hittable_type::hittable_list, (void*)&hittable_lists[0]};

    cam.render(&world);

    // === Cleanup ===
    delete[] glasses;
    delete[] metals;
    delete[] lambertians;
    delete[] lights;
    delete[] materials;
    delete[] quads;
    delete[] spheres;
    delete[] objects;
    delete[] hittable_lists;
}


void corridor() {
    // === Camera Setup ===
    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = WIDTH;
    cam.samples_per_pixel = SAMPLES;
    cam.max_depth = DEPTH;
    cam.background = color(0.0, 0.0, 0.0);
    cam.vfov = 35;
    cam.lookfrom = point3(0, 1, 3);
    cam.lookat = point3(0, 1, -5);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 0;

    // === World Structure ===
    const int max_glass = 2;
    const int max_metals = 2;
    const int max_light = 2;
    const int max_lambertians = 1;
    const int max_quad = 5;
    const int max_spheres = 2;
    const int max_materials = 8;
    const int max_objects = 11;
    const int max_hittable_list = 1;

    dielectric* glasses = new dielectric[max_glass];
    metal* metals = new metal[max_metals];
    lambertian* lambertians = new lambertian[max_lambertians];
    diffuse_light* lights = new diffuse_light[max_light];
    material* materials = new material[max_materials];
    quad* quads = new quad[max_quad];
    gpu_sphere* spheres = new gpu_sphere[max_spheres];
    hittable* objects = new hittable[max_objects];
    gpu_hittable_list* hittable_lists = new gpu_hittable_list[max_hittable_list];

    int glass_count = 0;
    int metal_count = 0;
    int light_count = 0;
    int lambertian_count = 0;
    int material_count = 0;
    int sphere_count = 0;
    int quad_count = 0;
    int object_count = 0;
    int hittable_list_count = 0;

    // === Materials ===
    glasses[glass_count++] = dielectric{1.5};  // Glass walls
    metals[metal_count++] = metal{color(0.95, 0.95, 0.95), 0.01}; // Mirror
    metals[metal_count++] = metal{color(0.2, 0.2, 0.2), 0.3};     // Matte dark

    lambertians[lambertian_count++] = lambertian{color(0.2, 0.3, 0.8)}; // Orb core

    lights[light_count++] = diffuse_light{color(0.6, 0.3, 1.0)};
    lights[light_count++] = diffuse_light{color(1.0, 0.2, 1.0)};

    materials[material_count++] = material{material_type::dielectric, (void*)&glasses[0]};
    materials[material_count++] = material{material_type::metal, (void*)&metals[0]};
    materials[material_count++] = material{material_type::metal, (void*)&metals[1]};
    materials[material_count++] = material{material_type::lambertian, (void*)&lambertians[0]};
    materials[material_count++] = material{material_type::diffuse_light, (void*)&lights[0]};
    materials[material_count++] = material{material_type::diffuse_light, (void*)&lights[1]};

    // === Spheres ===
    // Floating central orb
    spheres[sphere_count++] = gpu_sphere(point3(0, 1.0, -5), 0.5, &materials[3]);

    // Optional floor embed orb
    spheres[sphere_count++] = gpu_sphere(point3(0.7, 0.5, -6.0), 0.3, &materials[0]);

    for (int i = 0; i < sphere_count && object_count < max_objects; i++) {
        objects[object_count++] = hittable{hittable_type::sphere, (void*)&spheres[i]};
    }

    // === Quads (walls and lights) ===
    quads[quad_count++] = quad(point3(-2.0, 0, -10.0), vec3(0, 2.5, 0), vec3(0, 0, 10), &materials[1]); // Left glass wall
    quads[quad_count++] = quad(point3(2.0, 0, -10.0), vec3(0, 2.5, 0), vec3(0, 0, 10), &materials[1]);  // Right glass wall

    quads[quad_count++] = quad(point3(-1.5, 2.5, -10.0), vec3(3.0, 0, 0), vec3(0, 0, 10), &materials[1]); // Ceiling mirror
    quads[quad_count++] = quad(point3(-1.5, 0, -10.0), vec3(3.0, 0, 0), vec3(0, 0, 10), &materials[1]);   // Floor mirror

    quads[quad_count++] = quad(point3(-2, 0, -15), vec3(4, 0, 0), vec3(0, 3, 0), &materials[4]); // Bright white panel


    for (int i = 0; i < quad_count && object_count < max_objects; i++) {
        objects[object_count++] = hittable{hittable_type::quad, (void*)&quads[i]};
    }

    // === Assemble and Render ===
    hittable_lists[hittable_list_count++] = gpu_hittable_list{objects, object_count};
    hittable world = hittable{hittable_type::hittable_list, (void*)&hittable_lists[0]};
    cam.render(&world);

    // === Cleanup ===
    delete[] glasses;
    delete[] metals;
    delete[] lambertians;
    delete[] lights;
    delete[] materials;
    delete[] quads;
    delete[] spheres;
    delete[] objects;
    delete[] hittable_lists;
}






int main(int argc, char** argv) {
    switch (6) {
        case 1: {
            MPI_Init(&argc, &argv);
            mpi();
            break;
        } 
        case 2: cpu();      break;
        case 3: whole_image();  break;
        case 4: light();    break;
        case 5: floating(); break;
        case 6: corridor(); break;
>>>>>>> mpi
    }

    return 0;
}
