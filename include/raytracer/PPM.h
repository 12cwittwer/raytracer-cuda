#ifndef PPM_H
#define PPM_H

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "interval.h"
#include "vec3.h"

class PPM {
public:
    PPM() : mHeight(0), mWidth(0) {}  // Default constructor

    PPM(const int& height, const int& width)
        : mHeight(height), mWidth(width) {
        pixels.resize(height * width);
    }

    void setPixel(const int& row, const int& column, const vec3& pixel_color) {
        int i = index(row, column);

        pixels[i] = pixel_color;
    }

    int index(const int& row, const int& column) const {
        return (row * mWidth + column);
    }

    void writeImage() const {
        std::string filename = "img.ppm";
        std::ofstream file_out(filename);  // ASCII mode, not binary

        if (!file_out) {
            std::cerr << "Error: Unable to open file for writing\n";
            return;
        }

        writeStream(file_out);
        file_out.close();
    }

    void writeStream(std::ostream& os) const {
        os << "P3\n" << mWidth << ' ' << mHeight << "\n255\n";
        
        for (size_t i = 0; i < pixels.size(); i++) {
            write_color(os, pixels[i]);
        }
    }

private:
    std::vector<color> pixels;
    int mHeight, mWidth;
};

#endif // PPM_H