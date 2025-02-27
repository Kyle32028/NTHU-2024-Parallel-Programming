#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <time.h>

// #define M_PI 3.14159265358979323846f
#define T 1024
#define TILE 32
#define BLOCK 16

// Macro functions
#define ceil(a, b) ((a + b - 1) / b)
#define min(a, b) ((a < b)? a: b)

//----------------------------------------------
// Complex number operations for cuFloatComplex
//----------------------------------------------
__device__ inline cuFloatComplex multiplyComplex(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

__device__ inline cuFloatComplex addComplex(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x + b.x, a.y + b.y);
}

__device__ inline cuFloatComplex subtractComplex(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x - b.x, a.y - b.y);
}

__device__ inline float absComplex(cuFloatComplex a) {
    return sqrtf(a.x * a.x + a.y * a.y);
}

//----------------------------------------------
// Utility functions (host-side)
//----------------------------------------------
inline int power_of_two(int n) {
    int i = 1;
    while (i < n) i <<= 1;
    return i;
}

inline int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width, unsigned* channels) {
    unsigned char sig[8];
    FILE* infile = fopen(filename, "rb");
    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8)) return 1; // bad signature

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return 4; // out of memory
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4; // out of memory
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32 rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    *image = (unsigned char *) malloc(rowbytes * *height);
    png_bytep row_pointers[*height];

    #pragma omp parallel for
    for (unsigned i = 0; i < *height; ++i)
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

    // Grayscale output
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_FILTER_NONE);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);

    png_bytep row_ptr[height];

    #pragma omp parallel for
    for (int i = 0; i < (int)height; ++i)
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void generate_bit_reversal_indices(int* bit_reversed_indices, int n) {
    int log2_n = (int)log2f((float)n);
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int reversed = 0;
        int value = i;
        for (int j = 0; j < log2_n; j++) {
            reversed = (reversed << 1) | (value & 1);
            value >>= 1;
        }
        bit_reversed_indices[i] = reversed;
    }
}

//----------------------------------------------
// CUDA kernels
//----------------------------------------------

// Convert RGB to Gray
__global__ void convert_to_gray_kernel(unsigned char *s, unsigned char *t, int height, int width, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        int pos = idy * width + idx;
        int pos_rgb = idy * width * channels + idx * channels;
        int R = s[pos_rgb + 2];
        int G = s[pos_rgb + 1];
        int B = s[pos_rgb + 0];
        t[pos] = (unsigned char)(0.299f * R + 0.587f * G + 0.114f * B);
    }
}

// Initialize FFT input data (pad to power-of-two size)
__global__ void init_fft_data(unsigned char* gray, cuFloatComplex* fft_data,
                              int height, int width, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        fft_data[idy * w + idx] = make_cuFloatComplex((float)gray[idy * width + idx], 0.0f);
    } else {
        fft_data[idy * w + idx] = make_cuFloatComplex(0.0f, 0.0f);
    }
}

// Bit reversal reorder for all rows
__global__ void bit_reversal_reorder_kernel_all(
    cuFloatComplex* input, cuFloatComplex* output, int* bit_reversed_indices, int width, int height) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = ceil(height, gridDim.y);
    int sid = size * blockIdx.y;
    int end = min(sid + size, height);

    if (idx < width) {
        int reversed_idx = bit_reversed_indices[idx];
        for (int i = sid; i < end; ++i) {
            output[reversed_idx + i * width] = input[idx + i * width];
        }
    }
}

// Compute FFT row-wise (all rows in one kernel call)
__global__ void fft_rows_kernel_all(cuFloatComplex* data, int width) {
    int tid = threadIdx.x;
    int rid = blockIdx.x;
    int log2_n = (int)log2f((float)width);

    // Iterative FFT stage
    for (int stage = 1; stage <= log2_n; ++stage) {
        int m = 1 << stage;
        int m2 = m >> 1;
        for (int id = tid; id < width / 2; id += blockDim.x) {
            int group = id / m2;
            int pair = id % m2;
            int pos = rid * width + group * m + pair;

            float angle = -2.0f * M_PI * (float)pair / (float)m;
            cuFloatComplex w = make_cuFloatComplex(cosf(angle), sinf(angle));

            cuFloatComplex t = multiplyComplex(w, data[pos + m2]);
            cuFloatComplex u = data[pos];

            data[pos] = addComplex(u, t);
            data[pos + m2] = subtractComplex(u, t);
        }
        __syncthreads();
    }
}

__global__ void fft_rows_kernel_shared(cuFloatComplex* data, int w) {
    int row = blockIdx.x; 
    int tid = threadIdx.x;

    int ratio = (w + blockDim.x - 1) / blockDim.x;

    extern __shared__ cuFloatComplex s_data[];

    for (int i = 0; i < ratio; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < w) {
            s_data[idx] = data[row * w + idx];
        }
    }
    __syncthreads();

    int log2_n = (int)log2f((float)w);
    for (int stage = 1; stage <= log2_n; stage++) {
        int m = 1 << stage;
        int m2 = m >> 1;

        int pairs_total = w / 2;

        for (int i = 0; i < ratio; i++) {
            int pair_idx = tid + i * blockDim.x;
            if (pair_idx < pairs_total) {
                int group = pair_idx / m2;
                int pair = pair_idx % m2;
                int pos = group * m + pair;

                float angle = -2.0f * M_PI * (float)pair / (float)m;
                cuFloatComplex w_comp = make_cuFloatComplex(cosf(angle), sinf(angle));
                cuFloatComplex u = s_data[pos];
                cuFloatComplex t = multiplyComplex(w_comp, s_data[pos + m2]);

                s_data[pos] = addComplex(u, t);
                s_data[pos + m2] = subtractComplex(u, t);
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < ratio; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < w) {
            data[row * w + idx] = s_data[idx];
        }
    }
}

// Transpose using shared memory tile optimization
__global__ void transpose_kernel_tile(cuFloatComplex* input, cuFloatComplex* output, int height, int width) {
    __shared__ cuFloatComplex tile[TILE][TILE + 1];

    int idx = blockIdx.x * TILE + threadIdx.x;
    int idy = blockIdx.y * TILE + threadIdx.y;

    if (idy < height && idx < width)
        tile[threadIdx.y][threadIdx.x] = input[idy * width + idx];
    __syncthreads();

    int new_x = blockIdx.y * TILE + threadIdx.x;
    int new_y = blockIdx.x * TILE + threadIdx.y;

    if (new_y < width && new_x < height)
        output[new_y * height + new_x] = tile[threadIdx.x][threadIdx.y];
}

// Shift FFT data so the zero frequency is at the center
__global__ void center_frequency_kernel(cuFloatComplex* data, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    int half_h = height / 2;
    int half_w = width / 2;
    
    if (idx < half_w && idy < half_h) {
        int pos1 = idy * width + idx;
        int pos2 = (idy + half_h) * width + (idx + half_w);
        int pos3 = idy * width + (idx + half_w);
        int pos4 = (idy + half_h) * width + idx;
        
        cuFloatComplex temp1 = data[pos1];
        cuFloatComplex temp2 = data[pos3];
        
        data[pos1] = data[pos2];
        data[pos2] = temp1;
        data[pos3] = data[pos4];
        data[pos4] = temp2;
    }
}

// Compute max value (for normalization) in log scale
__global__ void compute_max_val_kernel(cuFloatComplex* data, float* max_val, int height, int width, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        float val = logf(1.0f + absComplex(data[idy * w + idx]));
        unsigned int val_int = __float_as_uint(val);
        atomicMax((unsigned int*)max_val, val_int);
    }
}

// Convert complex FFT data to image (0-255)
__global__ void convert_to_image_kernel(cuFloatComplex* data, unsigned char* output, 
                                        float* max_val, int height, int width, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        float maxv = __uint_as_float(*(unsigned int*)max_val);
        float val = logf(1.0f + absComplex(data[idy * w + idx]));
        output[idy * width + idx] = (unsigned char)((val / maxv) * 255.0f);
    }
}

//----------------------------------------------
// Host function to process image on GPU
//----------------------------------------------
void process_image_gpu(unsigned char* h_input, unsigned char* h_output, int height, int width, int channels) {
    // Allocate device memory
    unsigned char *d_input, *d_gray, *d_output;
    cuFloatComplex *d_fft_data, *d_transposed;
    float *d_max_val;
    int *d_indices;

    int w = power_of_two(width);
    int h = power_of_two(height);

    size_t input_size = height * width * channels * sizeof(unsigned char);
    size_t gray_size = height * width * sizeof(unsigned char);
    size_t complex_size = h * w * sizeof(cuFloatComplex);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_gray, gray_size);
    cudaMalloc(&d_fft_data, complex_size);
    cudaMalloc(&d_transposed, complex_size);
    cudaMalloc(&d_output, gray_size);
    cudaMalloc(&d_max_val, sizeof(float));
    cudaMalloc(&d_indices, max(w, h) * sizeof(int));

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Convert to gray
    convert_to_gray_kernel<<<gridDim, blockDim>>>(d_input, d_gray, height, width, channels);

    dim3 gridDimPad(w / blockDim.x, h / blockDim.y);
    init_fft_data<<<gridDimPad, blockDim>>>(d_gray, d_fft_data, height, width, w, h);

    int* h_indices = (int*)malloc(max(w, h) * sizeof(int));

    // FFT row-wise on padded data
    generate_bit_reversal_indices(h_indices, w);
    cudaMemcpy(d_indices, h_indices, w * sizeof(int), cudaMemcpyHostToDevice);

    bit_reversal_reorder_kernel_all<<<dim3(ceil(w, T), T), T>>>(d_fft_data, d_transposed, d_indices, w, h);
    cudaMemcpy(d_fft_data, d_transposed, complex_size, cudaMemcpyDeviceToDevice);

    if (w <= 4096) {
        int blockSize = 1024;
        int gridSize = h;
        size_t sharedMemSize = w * sizeof(cuFloatComplex);
        fft_rows_kernel_shared<<<gridSize, blockSize, sharedMemSize>>>(d_fft_data, w);
    } else {
        fft_rows_kernel_all<<<h, T>>>(d_fft_data, w);
    }

    // Transpose
    {
        int grid_x = (w + TILE - 1) / TILE;
        int grid_y = (h + TILE - 1) / TILE;
        transpose_kernel_tile<<<dim3(grid_x, grid_y), dim3(TILE, TILE)>>>(d_fft_data, d_transposed, h, w);
    }

    // FFT column-wise (now treated as rows after transpose)
    generate_bit_reversal_indices(h_indices, h);
    cudaMemcpy(d_indices, h_indices, h * sizeof(int), cudaMemcpyHostToDevice);

    bit_reversal_reorder_kernel_all<<<dim3(ceil(h, T), T), T>>>(d_transposed, d_fft_data, d_indices, h, w);
    cudaMemcpy(d_transposed, d_fft_data, complex_size, cudaMemcpyDeviceToDevice);

    if (h <= 4096) {
        int blockSize = 1024;
        int gridSize = w;
        size_t sharedMemSize = h * sizeof(cuFloatComplex);
        fft_rows_kernel_shared<<<gridSize, blockSize, sharedMemSize>>>(d_transposed, h);
    } else {
        fft_rows_kernel_all<<<w, T>>>(d_transposed, h);
    }

    // Transpose back
    {
        int grid_x = (h + TILE - 1) / TILE;
        int grid_y = (w + TILE - 1) / TILE;
        transpose_kernel_tile<<<dim3(grid_x, grid_y), dim3(TILE, TILE)>>>(d_transposed, d_fft_data, w, h);
    }

    // Center frequency and normalize to image
    cudaMemset(d_max_val, 0, sizeof(float));
    center_frequency_kernel<<<gridDimPad, blockDim>>>(d_fft_data, h, w);
    compute_max_val_kernel<<<gridDim, blockDim>>>(d_fft_data, d_max_val, height, width, w);
    convert_to_image_kernel<<<gridDim, blockDim>>>(d_fft_data, d_output, d_max_val, height, width, w);

    cudaMemcpy(h_output, d_output, gray_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_fft_data);
    cudaFree(d_transposed);
    cudaFree(d_output);
    cudaFree(d_max_val);
    cudaFree(d_indices);
    free(h_indices);
}

int main(int argc, char **argv) {
    struct timespec start, end;
    double time_used;
    
    clock_gettime(CLOCK_MONOTONIC, &start);

    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;

    // Read input image
    read_png(argv[1], &src, &height, &width, &channels);
    cudaMallocHost((void**)&dst, height * width * sizeof(unsigned char));

    // Process image on GPU
    process_image_gpu(src, dst, height, width, channels);

    // Write output image
    write_png(argv[2], dst, height, width, 1);

    // Clean up
    free(src);
    cudaFreeHost(dst);

    // End timing
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_used = (end.tv_sec - start.tv_sec) * 1000.0 +
                (end.tv_nsec - start.tv_nsec) / 1000000.0;
    printf("Total execution time: %.2f ms\n", time_used);
    return 0;
}
// 用openmp平行generate_bit_reversal_indices的迴圈

// 把init_fft_data跟center_frequency_kernel的grid設成
// dim3 gridDimPad(w / blockDim.x, h / blockDim.y);
// 原本是用width跟height，但這兩個kernel要處理的應該是padding後的，雖然不知道為什麼原本那樣不會錯，但感覺改成這樣比較安全

// 改成單精度後shared memory可以塞下6144個複數，所以我改成若長或寬小於4096，就用shared memory做fft，加速小測資的運算
