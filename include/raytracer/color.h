#ifndef COLOR_H
#define COLOR_H

#include "interval.h"
#include "vec3.h"

using color = vec3;

#ifndef __CUDA_ARCH__
#include <iostream>

inline double linear_to_gamma(double linear_component)
{
    if (linear_component > 0)
        return std::sqrt(linear_component);

    return 0;
}

// Clamp utility
inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

inline void write_color(std::ostream& out, const color& pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Apply a linear to gamma transform for gamma 2
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    // Clamp the values manually
    int rbyte = static_cast<int>(256 * clamp(r, 0.000, 0.999));
    int gbyte = static_cast<int>(256 * clamp(g, 0.000, 0.999));
    int bbyte = static_cast<int>(256 * clamp(b, 0.000, 0.999));

    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

inline void write_ppm(std::ostream& out, color* framebuffer, int height, int width) {
    // PPM header
    out << "P3\n" << width << " " << height << "\n255\n";

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            write_color(out, framebuffer[j * width + i]);
        }
    }

}
#endif

#endif
