#include <iostream>
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


    template <typename T>
    T* upload_array(const T* host_array, int count) {
        if (count == 0) return nullptr;
        T* device_array;
        CUDA_CHECK(cudaMalloc(&device_array, count * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(device_array, host_array, count * sizeof(T), cudaMemcpyHostToDevice));
        return device_array;
    }
    
    inline hittable* copy_scene_to_gpu(
        lambertian* lambertians, int lambertian_count,
        diffuse_light* lights, int light_count,
        material* materials, int material_count,
        gpu_sphere* spheres, int sphere_count,
        quad* quads, int quad_count,
        translate* translates, int translate_count,
        rotate_y* rotates, int rotate_count,
        gpu_hittable_list* lists, int list_count,
        hittable* objects, int object_count,
        bvh_node* nodes, int node_count,
        int root_index
    ) {
        // === Upload simple arrays ===
        lambertian* d_lambertians = upload_array(lambertians, lambertian_count);
        diffuse_light* d_lights = upload_array(lights, light_count);
        translate* d_translates = upload_array(translates, translate_count);
        rotate_y* d_rotates = upload_array(rotates, rotate_count);
        gpu_hittable_list* d_lists = upload_array(lists, list_count);
    
        // === Fix and upload materials ===
        material* fixed_materials = new material[material_count];
        for (int i = 0; i < material_count; ++i) {
            fixed_materials[i] = materials[i];
            if (materials[i].type == material_type::lambertian)
                fixed_materials[i].data = &d_lambertians[static_cast<lambertian*>(materials[i].data) - lambertians];
            else if (materials[i].type == material_type::diffuse_light)
                fixed_materials[i].data = &d_lights[static_cast<diffuse_light*>(materials[i].data) - lights];
        }
        material* d_materials = upload_array(fixed_materials, material_count);
        delete[] fixed_materials;
    
        // === Fix and upload spheres ===
        gpu_sphere* fixed_spheres = new gpu_sphere[sphere_count];
        for (int i = 0; i < sphere_count; ++i) {
            fixed_spheres[i] = spheres[i];
            fixed_spheres[i].mat_ptr = &d_materials[spheres[i].mat_ptr - materials];
        }
        gpu_sphere* d_spheres = upload_array(fixed_spheres, sphere_count);
        delete[] fixed_spheres;
    
        // === Fix and upload quads ===
        quad* fixed_quads = new quad[quad_count];
        for (int i = 0; i < quad_count; ++i) {
            fixed_quads[i] = quads[i];
            fixed_quads[i].mat_ptr = &d_materials[quads[i].mat_ptr - materials];
        }
        quad* d_quads = upload_array(fixed_quads, quad_count);
        delete[] fixed_quads;
    
        // === Fix and upload BVH nodes ===
        bvh_node* fixed_nodes = new bvh_node[node_count];
        for (int i = 0; i < node_count; ++i) {
            fixed_nodes[i] = nodes[i];
            auto fix = [&](hittable& h) {
                if (!h.data) return;
                switch (h.type) {
                    case hittable_type::sphere: h.data = &d_spheres[static_cast<gpu_sphere*>(h.data) - spheres]; break;
                    case hittable_type::quad: h.data = &d_quads[static_cast<quad*>(h.data) - quads]; break;
                    case hittable_type::bvh_node: h.data = &fixed_nodes[static_cast<bvh_node*>(h.data) - nodes]; break;
                    case hittable_type::translate: h.data = &d_translates[static_cast<translate*>(h.data) - translates]; break;
                    case hittable_type::rotate_y: h.data = &d_rotates[static_cast<rotate_y*>(h.data) - rotates]; break;
                    case hittable_type::hittable_list: h.data = &d_lists[static_cast<gpu_hittable_list*>(h.data) - lists]; break;
                    default: h.data = nullptr; break;
                }
            };
            fix(fixed_nodes[i].left);
            fix(fixed_nodes[i].right);
        }
        bvh_node* d_nodes = upload_array(fixed_nodes, node_count);
        delete[] fixed_nodes;
    
        // === Fix and upload hittables ===
        hittable* fixed_objects = new hittable[object_count];
        for (int i = 0; i < object_count; ++i) {
            fixed_objects[i] = objects[i];
            switch (objects[i].type) {
                case hittable_type::sphere: fixed_objects[i].data = &d_spheres[static_cast<gpu_sphere*>(objects[i].data) - spheres]; break;
                case hittable_type::quad: fixed_objects[i].data = &d_quads[static_cast<quad*>(objects[i].data) - quads]; break;
                case hittable_type::bvh_node: fixed_objects[i].data = &d_nodes[static_cast<bvh_node*>(objects[i].data) - nodes]; break;
                default: fixed_objects[i].data = nullptr; break;
            }
        }
        hittable* d_objects = upload_array(fixed_objects, object_count);
        delete[] fixed_objects;
    
        // === Determine Root Node Properly ===
        void* gpu_data = nullptr;
        switch (objects[root_index].type) {
            case hittable_type::sphere: gpu_data = &d_spheres[static_cast<gpu_sphere*>(objects[root_index].data) - spheres]; break;
            case hittable_type::quad: gpu_data = &d_quads[static_cast<quad*>(objects[root_index].data) - quads]; break;
            case hittable_type::bvh_node: gpu_data = &d_nodes[root_index]; break;
            case hittable_type::translate: gpu_data = &d_translates[static_cast<translate*>(objects[root_index].data) - translates]; break;
            case hittable_type::rotate_y: gpu_data = &d_rotates[static_cast<rotate_y*>(objects[root_index].data) - rotates]; break;
            case hittable_type::hittable_list: gpu_data = &d_lists[static_cast<gpu_hittable_list*>(objects[root_index].data) - lists]; break;
            default: gpu_data = nullptr; break;
        }
    
        hittable world_gpu = { objects[root_index].type, gpu_data };
        hittable* d_world = upload_array(&world_gpu, 1);
    
        return d_world;
    }



inline void create_box(
    const point3& a,
    const point3& b,
    material* mat,
    quad* quads,
    hittable* objects,
    int& quad_index,
    int& object_count
) {
    // Ensure min/max corners
    auto min = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
    auto max = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

    vec3 dx = vec3(max.x() - min.x(), 0, 0);
    vec3 dy = vec3(0, max.y() - min.y(), 0);
    vec3 dz = vec3(0, 0, max.z() - min.z());

    // +Z face (front)
    quads[quad_index] = { point3(min.x(), min.y(), max.z()), dx, dy, mat };
    objects[object_count] = { hittable_type::quad, &quads[quad_index] };
    ++object_count;
    ++quad_index;

    // +X face (right)
    quads[quad_index] = { point3(max.x(), min.y(), max.z()), -dz, dy, mat };
    objects[object_count] = { hittable_type::quad, &quads[quad_index] };
    ++object_count;
    ++quad_index;

    // -Z face (back)
    quads[quad_index] = { point3(max.x(), min.y(), min.z()), -dx, dy, mat };
    objects[object_count] = { hittable_type::quad, &quads[quad_index] };
    ++object_count;
    ++quad_index;

    // -X face (left)
    quads[quad_index] = { point3(min.x(), min.y(), min.z()), dz, dy, mat };
    objects[object_count] = { hittable_type::quad, &quads[quad_index] };
    ++object_count;
    ++quad_index;

    // +Y face (top)
    quads[quad_index] = { point3(min.x(), max.y(), max.z()), dx, -dz, mat };
    objects[object_count] = { hittable_type::quad, &quads[quad_index] };
    ++object_count;
    ++quad_index;

    // -Y face (bottom)
    quads[quad_index] = { point3(min.x(), min.y(), min.z()), dx, dz, mat };
    objects[object_count] = { hittable_type::quad, &quads[quad_index] };
    ++object_count;
    ++quad_index;
}

inline void create_transformed_box(
    const point3& min_corner,
    const point3& max_corner,
    material* mat,
    double rotate_angle,
    const vec3& translate_vec,
    quad* quads,
    hittable* objects,
    translate* translates,
    rotate_y* rotates,
    int& quad_index,
    int& object_index,
    int& rotate_index,
    int& translate_index
) {
    // Step 1: Create the box from 6 quads
    int base_index = object_index;
    create_box(min_corner, max_corner, mat, quads, objects, quad_index, object_index);

    // Step 2: Create rotate_y that wraps all 6 faces
    for (int i = 0; i < 6; ++i) {
        rotates[rotate_index] = rotate_y(&objects[base_index + i], rotate_angle);
        objects[base_index + i] = { hittable_type::rotate_y, &rotates[rotate_index++] };
    }

    // Step 3: Wrap each face in a translate
    for (int i = 0; i < 6; ++i) {
        translates[translate_index] = translate(&objects[base_index + i], translate_vec);
        objects[base_index + i] = { hittable_type::translate, &translates[translate_index++] };
    }
}


void spheres() {
    // === Storage capacities (tweakable) ===
    const int max_objects = 500;
    const int max_materials = max_objects + 5;

    // === Raw arrays ===
    lambertian* lambertians   = new lambertian[max_materials];
    metal* metals             = new metal[max_materials];
    dielectric* dielectrics   = new dielectric[max_materials];
    material* materials       = new material[max_materials];
    gpu_sphere* spheres       = new gpu_sphere[max_objects];
    hittable* objects         = new hittable[max_objects];

    int lambertian_count = 0;
    int metal_count = 0;
    int dielectric_count = 0;
    int material_count = 0;
    int sphere_count = 0;

    // === Ground ===
    lambertians[lambertian_count++] = { color(0.5, 0.5, 0.5) };
    materials[material_count++] = { material_type::lambertian, &lambertians[lambertian_count - 1] };
    spheres[sphere_count] = { point3(0, -1000, 0), 1000, &materials[material_count - 1] };
    objects[sphere_count] = { hittable_type::sphere, &spheres[sphere_count] };
    ++sphere_count;

    // === Randomized small spheres ===
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            if (sphere_count >= max_objects) continue;

            auto choose_mat = cpu_random_double();
            point3 center(a + 0.9 * cpu_random_double(), 0.2, b + 0.9 * cpu_random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material* mat_ptr = nullptr;

                if (choose_mat < 0.8) {
                    color albedo = random_vec3_host() * random_vec3_host();
                    lambertians[lambertian_count++] = { albedo };
                    materials[material_count++] = { material_type::lambertian, &lambertians[lambertian_count - 1] };
                } else if (choose_mat < 0.95) {
                    color albedo = random_vec3_host(0.5, 1.0);
                    double fuzz = cpu_random_double(0, 0.5);
                    metals[metal_count++] = { albedo, fuzz };
                    materials[material_count++] = { material_type::metal, &metals[metal_count - 1] };
                } else {
                    dielectrics[dielectric_count++] = { 1.5 };
                    materials[material_count++] = { material_type::dielectric, &dielectrics[dielectric_count - 1] };
                }

                spheres[sphere_count] = { center, 0.2, &materials[material_count - 1] };
                objects[sphere_count] = { hittable_type::sphere, &spheres[sphere_count] };
                ++sphere_count;
            }
        }
    }

    // === Big three spheres ===
    dielectrics[dielectric_count++] = { 1.5 };
    materials[material_count++] = { material_type::dielectric, &dielectrics[dielectric_count - 1] };
    spheres[sphere_count] = { point3(0, 1, 0), 1.0, &materials[material_count - 1] };
    objects[sphere_count] = { hittable_type::sphere, &spheres[sphere_count] };
    ++sphere_count;

    lambertians[lambertian_count++] = { color(0.4, 0.2, 0.1) };
    materials[material_count++] = { material_type::lambertian, &lambertians[lambertian_count - 1] };
    spheres[sphere_count] = { point3(-4, 1, 0), 1.0, &materials[material_count - 1] };
    objects[sphere_count] = { hittable_type::sphere, &spheres[sphere_count] };
    ++sphere_count;

    metals[metal_count++] = { color(0.7, 0.6, 0.5), 0.0 };
    materials[material_count++] = { material_type::metal, &metals[metal_count - 1] };
    spheres[sphere_count] = { point3(4, 1, 0), 1.0, &materials[material_count - 1] };
    objects[sphere_count] = { hittable_type::sphere, &spheres[sphere_count] };
    ++sphere_count;

    // === Build BVH ===
    bvh_node* nodes = new bvh_node[2 * sphere_count];
    int node_index = 0;
    int root_index = build_bvh(objects, 0, sphere_count, nodes, node_index);

    // Wrap root node as hittable
    hittable root = { hittable_type::bvh_node, &nodes[root_index] };

    // === Camera ===
    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 500;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;
    cam.background        = color(0.70, 0.80, 1.00);

    cam.vfov = 20;
    cam.lookfrom = point3(13, 2, 3);
    cam.lookat   = point3(0, 0, 0);
    cam.vup      = vec3(0, 1, 0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    // === Render ===
    cam.render_gpu(&root); // <- Pass BVH root hittable

    // === Clean up ===
    delete[] lambertians;
    delete[] metals;
    delete[] dielectrics;
    delete[] materials;
    delete[] spheres;
    delete[] objects;
    delete[] nodes;
}

void quads() {
    // === Storage caps ===
    const int quad_count = 5;
    lambertian* lambertians = new lambertian[quad_count];
    material* materials = new material[quad_count];
    quad* quads = new quad[quad_count];
    hittable* objects = new hittable[quad_count];

    // === Materials and Quads ===
    lambertians[0] = { color(1.0, 0.2, 0.2) };  // left red
    lambertians[1] = { color(0.2, 1.0, 0.2) };  // back green
    lambertians[2] = { color(0.2, 0.2, 1.0) };  // right blue
    lambertians[3] = { color(1.0, 0.5, 0.0) };  // upper orange
    lambertians[4] = { color(0.2, 0.8, 0.8) };  // lower teal

    for (int i = 0; i < quad_count; ++i) {
        materials[i] = { material_type::lambertian, &lambertians[i] };
    }

    quads[0] = { point3(-3,-2, 5), vec3(0, 0,-4), vec3(0, 4, 0), &materials[0] }; // left
    quads[1] = { point3(-2,-2, 0), vec3(4, 0, 0), vec3(0, 4, 0), &materials[1] }; // back
    quads[2] = { point3( 3,-2, 1), vec3(0, 0, 4), vec3(0, 4, 0), &materials[2] }; // right
    quads[3] = { point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), &materials[3] }; // top
    quads[4] = { point3(-2,-3, 5), vec3(4, 0, 0), vec3(0, 0,-4), &materials[4] }; // bottom

    for (int i = 0; i < quad_count; ++i) {
        objects[i] = { hittable_type::quad, &quads[i] };
    }

    // === BVH ===
    bvh_node* nodes = new bvh_node[2 * quad_count];
    int node_index = 0;
    int root_index = build_bvh(objects, 0, quad_count, nodes, node_index);
    hittable world = { hittable_type::bvh_node, &nodes[root_index]};
    // hittable root = { hittable_type::bvh_node, &nodes[root_index] };

    // === Camera ===
    camera cam;
    cam.aspect_ratio      = 1.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0.70, 0.80, 1.00);

    cam.vfov     = 80;
    cam.lookfrom = point3(0,0,9);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    cam.render(world);

    delete[] lambertians;
    delete[] materials;
    delete[] quads;
    delete[] objects;
    delete[] nodes;

}

void light() {
    // === Material storage ===
    lambertian* lambertians = new lambertian[3];
    diffuse_light* lights   = new diffuse_light[1];
    material* materials     = new material[3 + 1];  // 3 objects + 1 light
    gpu_sphere* spheres     = new gpu_sphere[2];
    hittable* objects       = new hittable[3];

    int lambertian_count = 0;
    int light_count = 0;
    int material_count = 0;
    int object_count = 0;

    // === Ground ===
    lambertians[lambertian_count++] = { color(0.4, 0.4, 0.4) };
    materials[material_count++] = { material_type::lambertian, &lambertians[lambertian_count - 1] };
    spheres[object_count] = { point3(0, -1000, 0), 1000.0, &materials[material_count - 1] };
    objects[object_count++] = { hittable_type::sphere, &spheres[object_count - 1] };

    // === Small Sphere ===
    lambertians[lambertian_count++] = { color(0.7, 0.2, 0.2) };
    materials[material_count++] = { material_type::lambertian, &lambertians[lambertian_count - 1] };
    spheres[object_count] = { point3(0, 2, 0), 2.0, &materials[material_count - 1] };
    objects[object_count++] = { hittable_type::sphere, &spheres[object_count - 1] };

    // === Diffuse Light ===
    lights[light_count++] = { color(4, 4, 4) };
    materials[material_count++] = { material_type::diffuse_light, &lights[light_count - 1] };

    // === Quad ===
    quad* quads = new quad[1];
    quads[0] = quad(point3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0), &materials[material_count - 1]);
    objects[object_count++] = { hittable_type::quad, &quads[0] };

    // === BVH ===
    bvh_node* nodes = new bvh_node[2 * object_count];
    int node_index = 0;
    int root_index = build_bvh(objects, 0, object_count, nodes, node_index);
    hittable world = { hittable_type::bvh_node, &nodes[root_index] };

    // === Camera Setup ===
    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;
    cam.background = color(0, 0, 0);

    cam.vfov = 20;
    cam.lookfrom = point3(26, 3, 6);
    cam.lookat   = point3(0, 2, 0);
    cam.vup      = vec3(0, 1, 0);

    cam.defocus_angle = 0;

    cam.render(world);

    // Cleanup
    delete[] lambertians;
    delete[] lights;
    delete[] materials;
    delete[] spheres;
    delete[] quads;
    delete[] objects;
}

void cornell() {
    const int max_quads = 100;
    const int max_objects = 200;
    
    quad* quads = new quad[max_quads];
    hittable* objects = new hittable[max_objects];
    translate* translates = new translate[2];
    rotate_y* rotates = new rotate_y[2];
    
    int quad_index = 0;
    int object_count = 0;

    // === Materials ===
    lambertian* lambertians = new lambertian[4];
    diffuse_light* lights = new diffuse_light[1];
    material* materials = new material[5];

    lambertians[0] = { color(.12, .45, .15) }; // green
    lambertians[1] = { color(.65, .05, .05) }; // red
    lambertians[2] = { color(.73, .73, .73) }; // white
    lights[0] = { color(15, 15, 15) };

    materials[0] = { material_type::lambertian, &lambertians[0] }; // green
    materials[1] = { material_type::lambertian, &lambertians[1] }; // red
    materials[2] = { material_type::lambertian, &lambertians[2] }; // white
    materials[3] = { material_type::diffuse_light, &lights[0] };

    // === Cornell walls ===
    quads[quad_index] = { point3(555,0,0), vec3(0,555,0), vec3(0,0,555), &materials[0] }; // green
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    quads[quad_index] = { point3(0,0,0), vec3(0,555,0), vec3(0,0,555), &materials[1] }; // red
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    quads[quad_index] = { point3(343,554,332), vec3(-130,0,0), vec3(0,0,-105), &materials[3] }; // light
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    quads[quad_index] = { point3(0,0,0), vec3(555,0,0), vec3(0,0,555), &materials[2] }; // floor
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    quads[quad_index] = { point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), &materials[2] }; // ceiling
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    quads[quad_index] = { point3(0,0,555), vec3(555,0,0), vec3(0,555,0), &materials[2] }; // back wall
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    // === Box 1 ===
    int box1_start = object_count;
    create_box(point3(0,0,0), point3(165,330,165), &materials[2], quads, objects, quad_index, object_count);

    gpu_hittable_list* box1_list = new gpu_hittable_list{ &objects[box1_start], 6 };
    hittable box1_group = { hittable_type::hittable_list, box1_list };

    rotates[0] = rotate_y(&box1_group, 15);
    hittable box1_rotated = { hittable_type::rotate_y, &rotates[0] };

    translates[0] = translate(&box1_rotated, vec3(265,0,295));
    objects[object_count++] = { hittable_type::translate, &translates[0] };

    // === Box 2 ===
    int box2_start = object_count;
    create_box(point3(0,0,0), point3(165,165,165), &materials[2], quads, objects, quad_index, object_count);

    gpu_hittable_list* box2_list = new gpu_hittable_list{ &objects[box2_start], 6 };
    hittable box2_group = { hittable_type::hittable_list, box2_list };

    rotates[1] = rotate_y(&box2_group, -18);
    hittable box2_rotated = { hittable_type::rotate_y, &rotates[1] };

    translates[1] = translate(&box2_rotated, vec3(130,0,65));
    objects[object_count++] = { hittable_type::translate, &translates[1] };


    // === BVH ===
    bvh_node* nodes = new bvh_node[2 * object_count];
    int node_index = 0;
    int root_index = build_bvh(objects, 0, object_count, nodes, node_index);
    hittable world = { hittable_type::bvh_node, &nodes[root_index] };

    // === Camera ===
    camera cam;
    cam.aspect_ratio = 1.0;
    cam.image_width = 600;
    cam.samples_per_pixel = 200;
    cam.max_depth = 50;
    cam.background = color(0,0,0);

    cam.vfov = 40;
    cam.lookfrom = point3(278, 278, -800);
    cam.lookat = point3(278, 278, 0);
    cam.vup = vec3(0,1,0);

    cam.defocus_angle = 0;

    cam.render(world);

    // === Cleanup ===
    delete[] quads;
    delete[] objects;
    delete[] lambertians;
    delete[] lights;
    delete[] materials;
    delete[] translates;
    delete[] rotates;
    delete[] nodes;
}

void final() {
    const int max_quads = 100;
    const int max_objects = 200;

    quad* quads = new quad[max_quads];
    hittable* objects = new hittable[max_objects];
    translate* translates = new translate[2];
    rotate_y* rotates = new rotate_y[2];
    gpu_sphere* spheres = new gpu_sphere[1];
    gpu_hittable_list* lists = new gpu_hittable_list[2];

    int quad_index = 0;
    int object_count = 0;

    // === Materials ===
    lambertian* lambertians = new lambertian[3];
    metal* metals = new metal[1];
    dielectric* dielectrics = new dielectric[1];
    diffuse_light* lights = new diffuse_light[1];
    material* materials = new material[6];

    lambertians[0] = { color(.12, .45, .15) }; // green
    lambertians[1] = { color(.65, .05, .05) }; // red
    lambertians[2] = { color(.73, .73, .73) }; // white
    metals[0] = { color(.8, .85, .88), 0.0 };  // mirror
    dielectrics[0] = { 1.5 };
    lights[0] = { color(15, 15, 15) };

    materials[0] = { material_type::lambertian, &lambertians[0] }; // green
    materials[1] = { material_type::lambertian, &lambertians[1] }; // red
    materials[2] = { material_type::lambertian, &lambertians[2] }; // white
    materials[3] = { material_type::diffuse_light, &lights[0] };
    materials[4] = { material_type::metal, &metals[0] };
    materials[5] = { material_type::dielectric, &dielectrics[0] };

    // === Cornell box walls ===
    quads[quad_index] = { point3(555,0,0), vec3(0,555,0), vec3(0,0,555), &materials[0] }; // green
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    quads[quad_index] = { point3(0,0,0), vec3(0,555,0), vec3(0,0,555), &materials[1] }; // red
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    quads[quad_index] = { point3(343,554,332), vec3(-130,0,0), vec3(0,0,-105), &materials[3] }; // light
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    quads[quad_index] = { point3(0,0,0), vec3(555,0,0), vec3(0,0,555), &materials[2] }; // floor
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    quads[quad_index] = { point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), &materials[2] }; // ceiling
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    quads[quad_index] = { point3(0,0,555), vec3(555,0,0), vec3(0,555,0), &materials[2] }; // back wall
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    // === Glass box ===
    int glass_start = object_count;
    create_box(point3(0,0,0), point3(165,330,165), &materials[5], quads, objects, quad_index, object_count);
    lists[0] = { &objects[glass_start], 6 };
    hittable glass_list = { hittable_type::hittable_list, &lists[0] };
    rotates[0] = rotate_y(&glass_list, 15);
    hittable glass_rot = { hittable_type::rotate_y, &rotates[0] };
    translates[0] = translate(&glass_rot, vec3(265,0,295));
    objects[object_count++] = { hittable_type::translate, &translates[0] };

    // === White box ===
    int box_start = object_count;
    create_box(point3(0,0,0), point3(165,165,165), &materials[2], quads, objects, quad_index, object_count);
    lists[1] = { &objects[box_start], 6 };
    hittable white_list = { hittable_type::hittable_list, &lists[1] };
    rotates[1] = rotate_y(&white_list, -18);
    hittable white_rot = { hittable_type::rotate_y, &rotates[1] };
    translates[1] = translate(&white_rot, vec3(130,0,65));
    objects[object_count++] = { hittable_type::translate, &translates[1] };

    // === Mirror sphere ===
    spheres[0] = { point3(278, 90, 160), 90.0, &materials[4] };
    objects[object_count++] = { hittable_type::sphere, &spheres[0] };

    // === BVH ===
    bvh_node* nodes = new bvh_node[2 * object_count];
    int node_index = 0;
    int root_index = build_bvh(objects, 0, object_count, nodes, node_index);
    hittable world = { hittable_type::bvh_node, &nodes[root_index] };

    // === Camera ===
    camera cam;
    cam.aspect_ratio = 1.0;
    cam.image_width = 300; // 600
    cam.samples_per_pixel = 100; // 400
    cam.max_depth = 20; // 50
    cam.background = color(0,0,0);
    cam.vfov = 40;
    cam.lookfrom = point3(278, 278, -800);
    cam.lookat = point3(278, 278, 0);
    cam.vup = vec3(0,1,0);
    cam.defocus_angle = 0;

    cam.render(world);

    // === Cleanup ===
    delete[] quads;
    delete[] objects;
    delete[] translates;
    delete[] rotates;
    delete[] lists;
    delete[] lambertians;
    delete[] metals;
    delete[] dielectrics;
    delete[] lights;
    delete[] materials;
    delete[] spheres;
    delete[] nodes;
}

void glass_box() {
    const int max_quads = 100;
    const int max_objects = 200;

    quad* quads = new quad[max_quads];
    hittable* objects = new hittable[max_objects];
    bvh_node* nodes = new bvh_node[2 * max_objects];
    lambertian* lambertians = new lambertian[3];
    diffuse_light* lights = new diffuse_light[1];
    dielectric* dielectrics = new dielectric[1];
    material* materials = new material[5];

    int quad_index = 0;
    int object_count = 0;
    int node_index = 0;

    // Materials
    lambertians[0] = { color(.73, .73, .73) }; // white walls
    lights[0] = { color(15, 15, 15) };         // light
    dielectrics[0] = { 1.5 };                  // glass

    materials[0] = { material_type::lambertian, &lambertians[0] };
    materials[1] = { material_type::diffuse_light, &lights[0] };
    materials[2] = { material_type::dielectric, &dielectrics[0] };

    // Floor, ceiling, back wall (white)
    quads[quad_index] = { point3(0,0,0), vec3(555,0,0), vec3(0,0,555), &materials[0] }; // floor
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    quads[quad_index] = { point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), &materials[0] }; // ceiling
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    quads[quad_index] = { point3(0,0,555), vec3(555,0,0), vec3(0,555,0), &materials[0] }; // back
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    // Light
    quads[quad_index] = { point3(213, 554, 227), vec3(130,0,0), vec3(0,0,105), &materials[1] };
    objects[object_count++] = { hittable_type::quad, &quads[quad_index++] };

    // Glass box - constructed as 6 quads in a BVH
    int glass_start = object_count;
    create_box(point3(0, 0, 0), point3(165, 165, 165), &materials[2], quads, objects, quad_index, object_count);
    
    // Build BVH over glass box faces
    bvh_node* glass_nodes = new bvh_node[12]; // 6 faces => max 2n-1 = 11, round up
    int glass_node_index = 0;
    int glass_root_index = build_bvh(objects, glass_start, object_count, glass_nodes, glass_node_index);

    hittable glass_box_bvh = { hittable_type::bvh_node, &glass_nodes[glass_root_index] };

    // Apply rotate + translate
    rotate_y* rotate_glass = new rotate_y[1];
    translate* translate_glass = new translate[1];

    rotate_glass[0] = rotate_y(&glass_box_bvh, 25);
    hittable glass_rotated = { hittable_type::rotate_y, &rotate_glass[0] };

    translate_glass[0] = translate(&glass_rotated, vec3(200, 0, 200));
    objects[object_count++] = { hittable_type::translate, &translate_glass[0] };

    // Final BVH for full scene
    int root_index = build_bvh(objects, 0, object_count, nodes, node_index);
    hittable world = { hittable_type::bvh_node, &nodes[root_index] };

    // Camera
    camera cam;
    cam.aspect_ratio = 1.0;
    cam.image_width = 600;
    cam.samples_per_pixel = 200;
    cam.max_depth = 50;
    cam.background = color(0,0,0);
    
    cam.vfov = 40;
    cam.lookfrom = point3(278, 278, -800);
    cam.lookat = point3(278, 278, 0);
    cam.vup = vec3(0,1,0);

    cam.defocus_angle = 0;

    cam.render(world);

    // Cleanup
    delete[] quads;
    delete[] objects;
    delete[] lambertians;
    delete[] lights;
    delete[] dielectrics;
    delete[] materials;
    delete[] nodes;
    delete[] glass_nodes;
    delete[] rotate_glass;
    delete[] translate_glass;
}

void glass_orb() {
    const int max_objects = 10;

    // === Storage ===
    lambertian* lambertians = new lambertian[2];
    dielectric* dielectrics = new dielectric[1];
    diffuse_light* lights = new diffuse_light[1];
    material* materials = new material[4];

    gpu_sphere* spheres = new gpu_sphere[3];
    hittable* objects = new hittable[max_objects];

    int object_count = 0;
    int material_index = 0;

    // === Ground ===
    lambertians[0] = { color(0.5, 0.5, 0.5) };
    materials[material_index++] = { material_type::lambertian, &lambertians[0] };

    spheres[object_count] = { point3(0, -1000, 0), 1000, &materials[material_index - 1] };
    objects[object_count++] = { hittable_type::sphere, &spheres[object_count] };

    // === Inner solid color sphere ===
    lambertians[1] = { color(1.0, 0.1, 0.1) };  // red
    materials[material_index++] = { material_type::lambertian, &lambertians[1] };

    spheres[object_count] = { point3(0, 1, 0), 0.5, &materials[material_index - 1] };
    objects[object_count++] = { hittable_type::sphere, &spheres[object_count] };

    // === Outer glass sphere ===
    dielectrics[0] = { 1.5 };
    materials[material_index++] = { material_type::dielectric, &dielectrics[0] };

    spheres[object_count] = { point3(0, 1, 0), 1.0, &materials[material_index - 1] };
    objects[object_count++] = { hittable_type::sphere, &spheres[object_count] };

    // === Optional: Light source above ===
    lights[0] = { color(10, 10, 10) };
    materials[material_index++] = { material_type::diffuse_light, &lights[0] };

    quad* quads = new quad[1];
    quads[0] = { point3(-2, 5, -2), vec3(4, 0, 0), vec3(0, 0, 4), &materials[material_index - 1] };
    objects[object_count++] = { hittable_type::quad, &quads[0] };

    // === BVH ===
    bvh_node* nodes = new bvh_node[2 * object_count];
    int node_index = 0;
    int root_index = build_bvh(objects, 0, object_count, nodes, node_index);
    hittable world = { hittable_type::bvh_node, &nodes[root_index] };

    // === Camera ===
    camera cam;
    cam.aspect_ratio = 1.0;
    cam.image_width = 600;
    cam.samples_per_pixel = 200;
    cam.max_depth = 100;
    cam.background = color(0,0,0);

    cam.vfov = 40;
    cam.lookfrom = point3(0, 2, -5);
    cam.lookat   = point3(0, 1, 0);
    cam.vup      = vec3(0,1,0);
    cam.defocus_angle = 0.0;

    cam.render(world);

    // === Cleanup ===
    delete[] lambertians;
    delete[] dielectrics;
    delete[] lights;
    delete[] materials;
    delete[] spheres;
    delete[] objects;
    delete[] nodes;
    delete[] quads;
}

void final_scene() {
    const int max_objects = 1000;
    const int max_spheres = 1000;
    const int max_quads = 10;
    const int max_transforms = 20;

    hittable* objects = new hittable[max_objects];
    gpu_sphere* spheres = new gpu_sphere[max_spheres];
    quad* quads = new quad[max_quads];
    translate* translates = new translate[max_transforms];
    rotate_y* rotates = new rotate_y[max_transforms];
    gpu_hittable_list* lists = new gpu_hittable_list[4];

    lambertian* lambertians = new lambertian[20];
    metal* metals = new metal[5];
    dielectric* dielectrics = new dielectric[5];
    diffuse_light* lights = new diffuse_light[2];
    material* materials = new material[30];

    int object_count = 0, sphere_count = 0, quad_count = 0;
    int lambertian_count = 0, metal_count = 0, dielectric_count = 0, light_count = 0, material_count = 0;
    int translate_count = 0, rotate_count = 0, list_count = 0;

    // === Ground Boxes ===
    lambertians[lambertian_count] = { color(0.48, 0.83, 0.53) };
    materials[material_count] = { material_type::lambertian, &lambertians[lambertian_count++] };
    material* ground_mat = &materials[material_count++];

    int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            if (object_count + 6 >= max_objects) continue;

            auto w = 100.0;
            auto x0 = -1000.0 + i * w;
            auto z0 = -1000.0 + j * w;
            auto y1 = cpu_random_double(1, 101);

            create_box(point3(x0, 0, z0), point3(x0 + w, y1, z0 + w), ground_mat,
                       quads, objects, quad_count, object_count);
        }
    }

    // === Light ===
    lights[light_count] = { color(7, 7, 7) };
    materials[material_count] = { material_type::diffuse_light, &lights[light_count++] };
    quads[quad_count] = { point3(123,554,147), vec3(300,0,0), vec3(0,0,265), &materials[material_count++] };
    objects[object_count++] = { hittable_type::quad, &quads[quad_count++] };

    // === Glass Sphere ===
    dielectrics[dielectric_count] = { 1.5 };
    materials[material_count] = { material_type::dielectric, &dielectrics[dielectric_count++] };
    spheres[sphere_count] = { point3(360,150,145), 70, &materials[material_count++] };
    objects[object_count++] = { hittable_type::sphere, &spheres[sphere_count++] };

    // === Metal Sphere Inside Glass ===
    metals[metal_count] = { color(0.8, 0.3, 0.1), 0.0 };
    materials[material_count] = { material_type::metal, &metals[metal_count++] };
    spheres[sphere_count] = { point3(360,150,145), 40, &materials[material_count++] };
    objects[object_count++] = { hittable_type::sphere, &spheres[sphere_count++] };

    // === Another metal sphere ===
    metals[metal_count] = { color(0.8, 0.8, 0.9), 1.0 };
    materials[material_count] = { material_type::metal, &metals[metal_count++] };
    spheres[sphere_count] = { point3(0, 150, 145), 50, &materials[material_count++] };
    objects[object_count++] = { hittable_type::sphere, &spheres[sphere_count++] };

    // === Sphere cluster ===
    lambertians[lambertian_count] = { color(.73, .73, .73) };
    materials[material_count] = { material_type::lambertian, &lambertians[lambertian_count++] };
    material* white = &materials[material_count++];

    int cluster_start = object_count;
    for (int j = 0; j < 1000 && object_count < max_objects; j++) {
        point3 center = random_vec3_host(0, 165);
        spheres[sphere_count] = { center, 10, white };
        objects[object_count++] = { hittable_type::sphere, &spheres[sphere_count++] };
    }

    lists[list_count] = { &objects[cluster_start], object_count - cluster_start };
    hittable list = { hittable_type::hittable_list, &lists[list_count++] };

    rotates[rotate_count] = rotate_y(&list, 15);
    hittable rotated = { hittable_type::rotate_y, &rotates[rotate_count++] };

    translates[translate_count] = translate(&rotated, vec3(-100, 270, 395));
    objects[object_count++] = { hittable_type::translate, &translates[translate_count++] };

    // === Build BVH ===
    bvh_node* nodes = new bvh_node[2 * object_count];
    int node_index = 0;
    int root_index = build_bvh(objects, 0, object_count, nodes, node_index);
    hittable world = { hittable_type::bvh_node, &nodes[root_index] };

    // === Camera ===
    camera cam;
    cam.aspect_ratio = 1.0;
    cam.image_width = 600;
    cam.samples_per_pixel = 200;
    cam.max_depth = 50;
    cam.background = color(0,0,0);

    cam.vfov = 40;
    cam.lookfrom = point3(478, 278, -600);
    cam.lookat = point3(278, 278, 0);
    cam.vup = vec3(0,1,0);
    cam.defocus_angle = 0;

    cam.render(world);

    // Cleanup
    delete[] objects;
    delete[] spheres;
    delete[] quads;
    delete[] translates;
    delete[] rotates;
    delete[] lists;
    delete[] lambertians;
    delete[] metals;
    delete[] dielectrics;
    delete[] lights;
    delete[] materials;
    delete[] nodes;
}

void my_scene() {
    // === Max counts for arrays ===
    const int max_lambertians = 3;
    const int max_lights = 1;
    const int max_materials = max_lambertians + max_lights;
    const int max_spheres = 2;
    const int max_quads = 1;
    const int max_objects = max_spheres + max_quads;
    const int max_nodes = 2 * max_objects; // safe upper bound for BVH
    const int max_translates = 0;
    const int max_rotates = 0;
    const int max_lists = 0;

    // === Host-side arrays ===
    lambertian* lambertians = new lambertian[max_lambertians];
    diffuse_light* lights = new diffuse_light[max_lights];
    material* materials = new material[max_materials];
    gpu_sphere* spheres = new gpu_sphere[max_spheres];
    quad* quads = new quad[max_quads];
    hittable* objects = new hittable[max_objects];
    bvh_node* nodes = new bvh_node[max_nodes];

    translate* translates = nullptr;
    rotate_y* rotates = nullptr;
    gpu_hittable_list* lists = nullptr;

    int lambertian_count = 0;
    int light_count = 0;
    int material_count = 0;
    int sphere_count = 0;
    int quad_count = 0;
    int object_count = 0;

    // === Build Scene ===

    // Ground Sphere
    lambertians[lambertian_count++] = { color(0.4, 0.4, 0.4) };
    materials[material_count++] = { material_type::lambertian, &lambertians[lambertian_count - 1] };
    spheres[sphere_count] = { point3(0, -1000, 0), 1000.0, &materials[material_count - 1] };
    objects[object_count++] = { hittable_type::sphere, &spheres[sphere_count++] };

    // Small Sphere
    lambertians[lambertian_count++] = { color(0.7, 0.2, 0.2) };
    materials[material_count++] = { material_type::lambertian, &lambertians[lambertian_count - 1] };
    spheres[sphere_count] = { point3(0, 2, 0), 2.0, &materials[material_count - 1] };
    objects[object_count++] = { hittable_type::sphere, &spheres[sphere_count++] };

    // Emissive Quad Light
    lights[light_count++] = { color(4, 4, 4) };
    materials[material_count++] = { material_type::diffuse_light, &lights[light_count - 1] };
    quads[quad_count] = { point3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0), &materials[material_count - 1] };
    objects[object_count++] = { hittable_type::quad, &quads[quad_count++] };

    // === BVH Build ===
    int node_index = 0;
    int root_index = build_bvh(objects, 0, object_count, nodes, node_index);

    // === Camera Setup ===
    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;
    cam.background = color(20, 20, 20);

    cam.vfov = 20;
    cam.lookfrom = point3(26, 3, 6);
    cam.lookat = point3(0, 2, 0);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 0;

    // === Upload Scene to GPU ===
    hittable* d_world = copy_scene_to_gpu(
        lambertians, lambertian_count,
        lights, light_count,
        materials, material_count,
        spheres, sphere_count,
        quads, quad_count,
        translates, max_translates,
        rotates, max_rotates,
        lists, max_lists,
        objects, object_count,
        nodes, node_index, // nodes actually created
        root_index
    );

    // === Render ===
    cam.render_gpu(d_world);

    // === Cleanup ===
    delete[] lambertians;
    delete[] lights;
    delete[] materials;
    delete[] spheres;
    delete[] quads;
    delete[] objects;
    delete[] nodes;
}

void smoll_boi() {
    const int max_lambertians = 1;
    const int max_spheres = 1;
    const int max_materials = 1;
    const int max_objects = 1;

    lambertian* lambertians = new lambertian[max_lambertians];
    material* materials = new material[max_materials];
    gpu_sphere* spheres = new gpu_sphere[max_spheres];
    hittable* objects = new hittable[max_objects];

    int lambertian_count = 0;
    int material_count = 0;
    int sphere_count = 0;
    int object_count = 0;

    // === Create Host Objects ===
    lambertians[lambertian_count++] = lambertian(color(0.4, 0.4, 0.4));
    materials[material_count++] = material(material_type::lambertian, nullptr); // Fix ptr later
    spheres[sphere_count++] = gpu_sphere(point3(0, -1000, 0), 50, nullptr);    // Fix ptr later
    objects[object_count++] = hittable(hittable_type::sphere, nullptr);         // Fix ptr later

    // === GPU Memory ===
    lambertian* d_lambertians;
    material* d_materials;
    gpu_sphere* d_spheres;
    hittable* d_objects;

    CUDA_CHECK(cudaMalloc(&d_lambertians, lambertian_count * sizeof(lambertian)));
    CUDA_CHECK(cudaMalloc(&d_materials, material_count * sizeof(material)));
    CUDA_CHECK(cudaMalloc(&d_spheres, sphere_count * sizeof(gpu_sphere)));
    CUDA_CHECK(cudaMalloc(&d_objects, object_count * sizeof(hittable)));

    CUDA_CHECK(cudaMemcpy(d_lambertians, lambertians, lambertian_count * sizeof(lambertian), cudaMemcpyHostToDevice));

    // === Fix Pointers Now that d_lambertians exists ===
    materials[0].ptr = d_lambertians;
    spheres[0].mat = d_materials;
    objects[0].ptr = d_spheres;

    // Copy fixed versions
    CUDA_CHECK(cudaMemcpy(d_materials, materials, material_count * sizeof(material), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spheres, spheres, sphere_count * sizeof(gpu_sphere), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_objects, objects, object_count * sizeof(hittable), cudaMemcpyHostToDevice));

    // === Camera Setup ===
    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;
    cam.background = color(20, 20, 20);
    cam.vfov = 20;
    cam.lookfrom = point3(26, 3, 6);
    cam.lookat = point3(0, 2, 0);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 0;

    // === Render ===
    cam.render_gpu(d_objects);
}

int main() {
    switch (10) {
        case 1: spheres();      break;
        case 2: quads();        break;
        case 3: light();        break;
        case 4: cornell();      break;
        case 5: final();        break;
        case 6: glass_box();    break;
        case 7: glass_orb();    break;
        case 8: final_scene();  break; // Seg Fault
        case 9: my_scene();     break;
        case 10: smoll_boi();   break;
    }

    return 0;
}
