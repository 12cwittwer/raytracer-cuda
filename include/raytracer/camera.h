#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "hittable_list.h"
#include "random_utils.h"
#include "material.h"
#include "hittable_dispatch_impl.h"
#include "camera_data.h"
#include "render.h"
#include "raytracer/cuda_utils.h"
#include "raytracer/PPM.h"

#include <mpi.h>
#include <vector>
#include <chrono>

extern void launch_render_kernel(const camera_data*, const hittable*, color*, int, int, int);

const int TAG_REQUEST = 1, TAG_WORK = 2, TAG_RESULT = 3, TAG_STOP = 4;

class camera {
  public:
    double aspect_ratio = 1.0;  // Ratio of image width over height
    int    image_width  = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10; // Count of random samples for each pixel
    int    max_depth         = 10;   // Maximum number of ray bounces into scene
    color  background;

    double vfov = 90; // Vertical view angle (field of view)
    point3 lookfrom = point3(0,0,0);
    point3 lookat = point3(0,0,-1);
    vec3 vup = vec3(0,1,0);

    double defocus_angle = 0;
    double focus_dist = 10;

    void render_gpu(const hittable* d_world, const int rank, const int num_procs) {
        initialize();
    
        color* d_fb = nullptr;
        CUDA_CHECK(cudaMalloc(&d_fb, image_width * sizeof(color)));
    
        camera_data h_cam = get_camera_data();
    
        camera_data* d_cam = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cam, sizeof(camera_data)));
        CUDA_CHECK(cudaMemcpy(d_cam, &h_cam, sizeof(camera_data), cudaMemcpyHostToDevice));
    
        while (true) {
            int row;
            MPI_Status status;
    
            MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_STOP) break;
    
            launch_render_kernel(d_cam, d_world, d_fb, image_width, image_height, row);
    
            std::vector<color> h_fb(image_width);
            CUDA_CHECK(cudaMemcpy(h_fb.data(), d_fb, image_width * sizeof(color), cudaMemcpyDeviceToHost));
    
            std::vector<float> buffer(1 + image_width * 3); // 1 for row index
            buffer[0] = static_cast<float>(row);
            for (int i = 0; i < image_width; i++) {
                buffer[1 + i * 3 + 0] = h_fb[i].x();
                buffer[1 + i * 3 + 1] = h_fb[i].y();
                buffer[1 + i * 3 + 2] = h_fb[i].z();
            }
            
            MPI_Send(buffer.data(), 1 + image_width * 3, MPI_FLOAT, 0, TAG_RESULT, MPI_COMM_WORLD);
            
        }
    
        CUDA_CHECK(cudaFree(d_fb));
        CUDA_CHECK(cudaFree(d_cam));
    }
    

    void delegate(const int rank, const int num_procs) { 
        initialize();
    
        auto start = std::chrono::high_resolution_clock::now();
    
        PPM image = PPM(image_height, image_width);
    
        int next_row = 0, active_workers = num_procs - 1;
        MPI_Status status;
    
        // Kick off initial work
        for (int i = 1; i < num_procs; i++) {
            if (next_row < image_height) {
                MPI_Send(&next_row, 1, MPI_INT, i, TAG_WORK, MPI_COMM_WORLD);
                next_row++;
            }
        }
    
        while (active_workers > 0) {
            std::vector<float> buffer(1 + image_width * 3);
            MPI_Recv(buffer.data(), 1 + image_width * 3, MPI_FLOAT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
            
            int row_index = static_cast<int>(buffer[0]);
            int worker_rank = status.MPI_SOURCE;
            
            for (int i = 0; i < image_width; i++) {
                color pixel_color(buffer[1 + i * 3 + 0],
                                  buffer[1 + i * 3 + 1],
                                  buffer[1 + i * 3 + 2]);
                image.setPixel(row_index, i, pixel_color);
            }
                   
    
            if (next_row < image_height) {
                MPI_Send(&next_row, 1, MPI_INT, worker_rank, TAG_WORK, MPI_COMM_WORLD);
                next_row++;
                std::cout << "\rScanlines remaining: " << (image_height - next_row) << " " << std::flush;
            } else {
                MPI_Send(nullptr, 0, MPI_INT, worker_rank, TAG_STOP, MPI_COMM_WORLD);
                active_workers--;
            }
        }
    
        image.writeImage();
    
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Execution Time: " << duration.count() / 1000.0 << " s\n";
    }
    
    

    void render(const hittable* world) {
        initialize();

        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = 0; j < image_height; j++) {
            for (int i = 0; i < image_width; i++) {
                color pixel_color(0, 0, 0);
                for (int sample = 0; sample < samples_per_pixel; sample++) {
                    ray r = get_ray(i, j);
                    pixel_color += ray_color(r, max_depth, *world);
                }
                write_color(std::cout, pixel_samples_scale * pixel_color);
            }
        }

        std::clog << "\rDone.                 \n";
    }

  private:
    int    image_height;   // Rendered image height
    double pixel_samples_scale;  // Color scale factor for a sum of pixel samples
    point3 center;         // Camera center
    point3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below
    vec3   u, v, w;
    vec3   defocus_disk_u;
    vec3   defocus_disk_v;

    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta/2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    ray get_ray(int i, int j) const {
        // Construct a camera ray originating from the origin and directed at randomly sampled
        // point around the pixel location i, j.

        auto offset = sample_square();
        auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    vec3 sample_square() const {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return vec3(cpu_random_double() - 0.5, cpu_random_double() - 0.5, 0);
    }

    point3 defocus_disk_sample() const {
        auto p = random_in_unit_disk_host();
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    color ray_color(const ray& r, int depth, const hittable& world) const {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0,0,0);

        hit_record rec;
        hit_hittable(world, r, interval(0.001, infinity), rec);

        if (!rec.hit) {
            return background;
        }

        ray scattered;
        color attenuation;
        color color_from_emission = emitted_material(*rec.mat_ptr, rec.u, rec.v, rec.p);

        if (!scatter_material_host(*rec.mat_ptr, r, rec, attenuation, scattered)) {
            return color_from_emission;
        }

        color color_from_scatter = attenuation * ray_color(scattered, depth - 1, world);

        return color_from_emission + color_from_scatter;
    }

    camera_data get_camera_data() const {
        camera_data cam;
    
        cam.center = center;
        cam.pixel00_loc = pixel00_loc;
        cam.pixel_delta_u = pixel_delta_u;
        cam.pixel_delta_v = pixel_delta_v;
        cam.defocus_disk_u = defocus_disk_u;
        cam.defocus_disk_v = defocus_disk_v;
        cam.background = background;
        cam.image_width = image_width;
        cam.image_height = image_height;
        cam.samples_per_pixel = samples_per_pixel;
        cam.max_depth = max_depth;
        cam.defocus_angle = static_cast<float>(defocus_angle);
        cam.pixel_samples_scale = static_cast<float>(pixel_samples_scale);
    
        return cam;
    }
    
};

#endif
