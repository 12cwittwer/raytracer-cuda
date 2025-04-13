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

void mpi() {
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // === Camera Setup ===
    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 800;
    cam.samples_per_pixel = 100;
    cam.max_depth = 10;
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
    cam.image_width = 800;
    cam.samples_per_pixel = 100;
    cam.max_depth = 10;
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
    cam.image_width = 800;
    cam.samples_per_pixel = 100;
    cam.max_depth = 10;
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



int main(int argc, char** argv) {
    switch (3) {
        case 1: {
            MPI_Init(&argc, &argv);
            mpi();
            break;
        } 
        case 2: cpu();      break;
        case 3: whole_image();  break;
    }

    return 0;
}
