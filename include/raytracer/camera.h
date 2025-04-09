#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "hittable_list.h"
#include "random_utils.h"
#include "material.h"
#include "hittable_dispatch_impl.h"
#include "camera_data.h"
#include "render.h"

extern void launch_render_kernel(const camera_data*, const hittable*, color*, int, int);

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

    void render_gpu(const hittable* d_world) {
        initialize();

        int image_size = image_width * image_height;

        // Allocate framebuffer
        color* fb_device;
        cudaMalloc(&fb_device, image_size * sizeof(color));

        // Copy camera data to GPU
        camera_data cam_data = get_camera_data();
        camera_data* cam_device;
        cudaMalloc(&cam_device, sizeof(camera_data));
        cudaMemcpy(cam_device, &cam_data, sizeof(camera_data), cudaMemcpyHostToDevice);

        // Copy world to GPU
        // hittable* world_device;
        // cudaMalloc(&world_device, sizeof(hittable));
        // cudaMemcpy(world_device, &world, sizeof(hittable), cudaMemcpyHostToDevice);

        // Launch CUDA render kernal
        launch_render_kernel(cam_device, d_world, fb_device, image_width, image_height);

        // Copy framebuffer back
        color* framebuffer = new color[image_size];
        cudaMemcpy(framebuffer, fb_device, image_size * sizeof(color), cudaMemcpyDeviceToHost);

        // Output Image
        write_ppm(std::cout, framebuffer, image_height, image_width);
        delete[] framebuffer;
    }

    void render(const hittable& world) {
        initialize();

        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = 0; j < image_height; j++) {
            std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
            for (int i = 0; i < image_width; i++) {
                color pixel_color(0, 0, 0);
                for (int sample = 0; sample < samples_per_pixel; sample++) {
                    ray r = get_ray(i, j);
                    pixel_color += ray_color(r, max_depth, world);
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
        rec = hit_hittable(world, r, interval(0.001, infinity), rec);

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
        cam.defocus_angle = defocus_angle;
    
        return cam;
    }
};

#endif
