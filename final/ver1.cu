#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <cuda.h>

#define M_PI 3.14159265358979323846

using namespace std;

typedef complex<double> ComplexD;
typedef vector<vector<ComplexD>> MatrixD;

inline int power_of_two(int n) {
    int i = 1;
    while (i < n)
        i <<= 1;
    return i;
}

inline int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

inline void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

inline void bit_reversal(vector<ComplexD>& data) {
    int n = data.size();
    for (int i = 1, j = 0; i < n; ++i) {
        for (int k = n >> 1; !((j ^= k) & k); k >>= 1);
        if (i > j) swap(data[i], data[j]);
    }
}

// FFT function (FFT)
inline void fft(vector<ComplexD>& data) {
    int n = data.size();

    bit_reversal(data);

    for (int k = 2; k <= n; k <<= 1) {
        double angle = -2.0 * M_PI / k;
        ComplexD delta_w(cos(angle), sin(angle));

        for (int j = 0; j < n; j += k) {
            ComplexD w(1, 0);
            for (int i = j; i < j + k / 2; i++) {
                ComplexD t = data[i + k / 2] * w;
                ComplexD u = data[i];
                data[i] = u + t;
                data[i + k / 2] = u - t;
                w *= delta_w;
            }
        }
    }
}

// Inverse FFT function (IFFT)
inline void ifft(vector<ComplexD>& a) {
    int n = a.size();

    for (int i = 0; i < n; ++i) {
        a[i] = conj(a[i]);
    }

    fft(a);

    for (int i = 0; i < n; ++i) {
        a[i] = conj(a[i]) / (double)n;
    }
}

inline void center_frequency(MatrixD& data, int height, int width) {
    int half_h = height / 2;
    int half_w = width / 2;

    for (int i = 0; i < half_h; ++i) {
        for (int j = 0; j < half_w; ++j) {
            // Swap quadrants
            swap(data[i][j], data[i + half_h][j + half_w]);
            swap(data[i][j + half_w], data[i + half_h][j]);
        }
    }
}

void ifft_process_2d(MatrixD& data, int height, int width) {
    int h = data.size();
    int w = data[0].size();
   
    vector<ComplexD> cols(h);
    for (int j = 0; j < w; ++j) {
        for (int i = 0; i < h; ++i)
            cols[i] = data[i][j];
        ifft(cols);
        for (int i = 0; i < h; ++i)
            data[i][j] = cols[i];
    }

    for (int i = 0; i < h; ++i)
        ifft(data[i]);
}

void fft_process_2d(unsigned char *s, unsigned char *t, int height, int width) {
    int w = power_of_two(width);
    int h = power_of_two(height);

    MatrixD data(h, vector<ComplexD>(w));

    // initilize matrix
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            data[i][j] = ComplexD((int)s[i * width + j], 0);
    for (int i = height; i < h; ++i)
        for (int j = width; j < w; ++j)
            data[i][j] = ComplexD(0, 0);

    // y-axis fft
    for (int i = 0; i < h; ++i)
        fft(data[i]);

    // x-axis fft
    vector<ComplexD> cols(h);
    for (int j = 0; j < w; ++j) {
        for (int i = 0; i < h; ++i) cols[i] = data[i][j];
        fft(cols);
        for (int i = 0; i < h; ++i) data[i][j] = cols[i];
    }

    center_frequency(data, h, w);

    // calculate magnitude
    double maxVal = 0;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            maxVal = max(maxVal, log(1 + abs(data[y][x])));

    for (int y = 0; y < height; ++y){
        for (int x = 0; x < width; ++x) {
            double val = log(1 + abs(data[y][x]));
            t[y * width + x] = static_cast<int>((val / maxVal) * 255);
        }
    }
    
}

// CUDA kernel for grayscale conversion
__global__ void convert_to_gray_kernel(unsigned char *s, unsigned char *t, int height, int width, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        int pos = idy * width + idx;
        int pos_rgb = idy * width * channels + idx * channels;
        
        int R = s[pos_rgb + 2];
        int G = s[pos_rgb + 1];
        int B = s[pos_rgb + 0];
        
        t[pos] = 0.299f * R + 0.587f * G + 0.114f * B;
    }
}

void convert_to_gray(unsigned char *s, unsigned char *t, int height, int width, int channels) {
    unsigned char *d_src, *d_dst;
    size_t size_src = height * width * channels * sizeof(unsigned char);
    size_t size_dst = height * width * sizeof(unsigned char);
    
    // Allocate device memory
    cudaMalloc(&d_src, size_src);
    cudaMalloc(&d_dst, size_dst);
    
    // Copy input image to device
    cudaMemcpy(d_src, s, size_src, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    dim3 blockDim(32, 32);
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y
    );
    
    // Launch kernel
    convert_to_gray_kernel<<<gridDim, blockDim>>>(d_src, d_dst, height, width, channels);
    
    // Copy result back to host
    cudaMemcpy(t, d_dst, size_dst, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dst);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

int main(int argc, char **argv) {

    unsigned height, width, channels;
    unsigned char *src = NULL, *gray, *dst;

    read_png(argv[1], &src, &height, &width, &channels);
    gray = (unsigned char *) malloc(height * width * sizeof(unsigned char));
    dst = (unsigned char *) malloc(height * width * sizeof(unsigned char));

    convert_to_gray(src, gray, height, width, channels);

    fft_process_2d(gray, dst, height, width);

    write_png(argv[2], dst, height, width, 1);

    free(src);
    free(gray);
    free(dst);
    return 0;
}