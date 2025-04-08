#ifndef MATERIAL_H
#define MATERIAL_H

#include "vec3.h"
#include "ray.h"
#include "hittable.h"
#include "random_utils.h"

enum class material_type {
    lambertian = 0,
    metal = 1,
    dielectric = 2,
    diffuse_light = 3
    // metal, dielectric, etc. to be added later
};

struct material {
    material_type type;
    void* data;
};

struct lambertian {
    color albedo;
};

struct metal {
    color albedo;
    double fuzz;
};

struct dielectric {
    double refraction_index;
};

struct diffuse_light {
    color emit;
};

__host__ __device__ static double reflectance(double cosine, double refraction_index) {
    auto r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__device__ inline bool scatter_material(
    const material& mat,
    const ray& r_in,
    const hit_record& rec,
    color& attenuation,
    ray& scattered,
    curandState* rng
) {
    switch (mat.type) {
        case material_type::lambertian: {
            const lambertian* lam = reinterpret_cast<const lambertian*>(mat.data);
            vec3 scatter_direction = rec.normal + random_unit_vector(rng);

            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;

            scattered = ray(rec.p, scatter_direction);
            attenuation = lam->albedo;

            return true;
        }
        case material_type::metal: {
            const metal* m = reinterpret_cast<const metal*>(mat.data);
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            reflected = unit_vector(reflected) + (m->fuzz * random_unit_vector(rng));
            scattered = ray(rec.p, reflected);
            attenuation = m->albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }
        case material_type::dielectric: {
            const dielectric* d = reinterpret_cast<const dielectric*>(mat.data);

            attenuation = color(1.0, 1.0, 1.0);
            double ri = rec.front_face ? (1.0 / d->refraction_index) : d->refraction_index;

            vec3 unit_direction = unit_vector(r_in.direction());
            double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            bool cannot_refract = ri * sin_theta > 1.0;
            vec3 direction;

            if (cannot_refract || reflectance(cos_theta, ri) > random_double(rng)) {
                direction = reflect(unit_direction, rec.normal);
            } else {
                direction = refract(unit_direction, rec.normal, ri);
            }

            scattered = ray(rec.p, direction);
            return true;
        }
        case material_type::diffuse_light: {
            return false;
        }
        default:
            return false;
    }
}

__host__ __device__
inline color emitted_material(const material& mat, double u, double v, const point3& p) {
    switch (mat.type) {
        case material_type::diffuse_light:
            return reinterpret_cast<const diffuse_light*>(mat.data)->emit;
        default:
            return color(0, 0, 0);
    }
}

__host__ inline bool scatter_material_host(
    const material& mat,
    const ray& r_in,
    const hit_record& rec,
    color& attenuation,
    ray& scattered
) {
    switch (mat.type) {
        case material_type::lambertian: {
            const lambertian* lam = reinterpret_cast<const lambertian*>(mat.data);
            vec3 scatter_direction = rec.normal + random_unit_vector_host();

            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;

            scattered = ray(rec.p, scatter_direction);
            attenuation = lam->albedo;

            return true;
        }
        case material_type::metal: {
            const metal* m = reinterpret_cast<const metal*>(mat.data);
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            reflected = unit_vector(reflected) + (m->fuzz * random_unit_vector_host());
            scattered = ray(rec.p, reflected);
            attenuation = m->albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }
        case material_type::dielectric: {
            const dielectric* d = reinterpret_cast<const dielectric*>(mat.data);

            attenuation = color(1.0, 1.0, 1.0);
            double ri = rec.front_face ? (1.0 / d->refraction_index) : d->refraction_index;

            vec3 unit_direction = unit_vector(r_in.direction());
            double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            bool cannot_refract = ri * sin_theta > 1.0;
            vec3 direction;

            if (cannot_refract || reflectance(cos_theta, ri) > random_double_host()) {
                direction = reflect(unit_direction, rec.normal);
            } else {
                direction = refract(unit_direction, rec.normal, ri);
            }

            scattered = ray(rec.p, direction);
            return true;
        }
        case material_type::diffuse_light: {
            return false;
        }
        default:
            return false;
    }
}

#endif
