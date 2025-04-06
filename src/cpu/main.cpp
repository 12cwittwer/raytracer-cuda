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

            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material* mat_ptr = nullptr;

                if (choose_mat < 0.8) {
                    color albedo = random_vec3() * random_vec3();
                    lambertians[lambertian_count++] = { albedo };
                    materials[material_count++] = { material_type::lambertian, &lambertians[lambertian_count - 1] };
                } else if (choose_mat < 0.95) {
                    color albedo = random_vec3(0.5, 1.0);
                    double fuzz = random_double(0, 0.5);
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
    cam.render(root); // <- Pass BVH root hittable

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
    cam.image_width = 600;
    cam.samples_per_pixel = 400;
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



int main() {
    switch (4) {
        case 1: spheres();  break;
        case 2: quads();    break;
        case 3: light();    break;
        case 4: cornell();  break;
        case 5: final();    break;
    }

    return 0;
}
