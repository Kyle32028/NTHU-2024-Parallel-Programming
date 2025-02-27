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

#define M_PI 3.14159265358979323846
#define T 1024
#define TILE 32
#define BLOCK 16
#define ceil(a, b) ((a + b - 1) / b)
#define min(a, b) ((a < b)? a: b)
// Helper functions for cuDoubleComplex operations
__device__ inline cuDoubleComplex multiplyComplex(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

__device__ inline cuDoubleComplex addComplex(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__device__ inline cuDoubleComplex subtractComplex(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
}

__device__ inline double absComplex(cuDoubleComplex a) {
    return sqrt(a.x * a.x + a.y * a.y);
}

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

void generate_bit_reversal_indices(int* bit_reversed_indices, int n) {
    int log2_n = log2(n);  // 假設 n 是 2 的冪次
    for (int i = 0; i < n; i++) {
        int reversed = 0;
        int value = i;
        for (int j = 0; j < log2_n; j++) {
            reversed = (reversed << 1) | (value & 1);  // 將每一位翻轉
            value >>= 1;  // 右移
        }
        bit_reversed_indices[i] = reversed;
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

__global__ void init_fft_data(unsigned char* gray, cuDoubleComplex* fft_data, 
                             int height, int width, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < w && idy < h) {
        if (idx < width && idy < height) {
            fft_data[idy * w + idx] = make_cuDoubleComplex((double)gray[idy * width + idx], 0);
        } else {
            fft_data[idy * w + idx] = make_cuDoubleComplex(0, 0);
        }
    }
}

__global__ void bit_reversal_reorder_kernel(
    cuDoubleComplex* input, cuDoubleComplex* output, int* bit_reversed_indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int reversed_idx = bit_reversed_indices[idx];
        output[reversed_idx] = input[idx];
    }
}

__global__ void bit_reversal_reorder_kernel_all(
    cuDoubleComplex* input, cuDoubleComplex* output, int* bit_reversed_indices, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = ceil(height, gridDim.y);
    int sid = size * blockIdx.y;
    int end = min(sid + size, height);
    /*if (!idx)
        printf("sid %d\nend %d\nheight %d\n", sid, end, height);*/
    //int end = min((height >> 1) * (blockIdx.y + 1), height);

    if (idx < width) {
        int reversed_idx = bit_reversed_indices[idx];
        for (int i = sid; i < end; ++i) {
            output[reversed_idx + i * width] = input[idx + i * width];
        }
    }
}


__global__ void fft_kernel_row(cuDoubleComplex* data, int n, int stage, int row, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = 1 << stage;
    int m2 = m >> 1;

    if (idx < n / 2) {
        int group = idx / m2;
        int pair = idx % m2;
        int pos = row * width + group * m + pair;

        double angle = -2.0 * M_PI * pair / m;
        cuDoubleComplex w = make_cuDoubleComplex(cos(angle), sin(angle));
        cuDoubleComplex t = multiplyComplex(w, data[pos + m2]);
        cuDoubleComplex u = data[pos];

        data[pos] = addComplex(u, t);
        data[pos + m2] = subtractComplex(u, t);
    }
}

__global__ void transpose_kernel(cuDoubleComplex* input, cuDoubleComplex* output, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        output[idx * height + idy] = input[idy * width + idx];
    }
}

__global__ void center_frequency_kernel(cuDoubleComplex* data, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    int half_h = height / 2;
    int half_w = width / 2;
    
    if (idx < half_w && idy < half_h) {
        int pos1 = idy * width + idx;
        int pos2 = (idy + half_h) * width + (idx + half_w);
        int pos3 = idy * width + (idx + half_w);
        int pos4 = (idy + half_h) * width + idx;
        
        cuDoubleComplex temp1 = data[pos1];
        cuDoubleComplex temp2 = data[pos3];
        
        data[pos1] = data[pos2];
        data[pos2] = temp1;
        data[pos3] = data[pos4];
        data[pos4] = temp2;
    }
}

__global__ void compute_max_val_kernel(cuDoubleComplex* data, double* max_val, int height, int width, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        double val = log(1 + absComplex(data[idy * w + idx]));
        atomicMax((unsigned long long*)max_val, __double_as_longlong(val));
    }
}

__global__ void convert_to_image_kernel(cuDoubleComplex* data, unsigned char* output, 
                                      double* max_val, int height, int width, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        double val = log(1 + absComplex(data[idy * w + idx]));
        output[idy * width + idx] = static_cast<unsigned char>((val / *max_val) * 255);
    }
}

// 單一 kernel 完成所有 rows 的 FFT 計算 (row-wise FFT)
__global__ void fft_rows_kernel(cuDoubleComplex* data, int* bit_reversed_indices, int width, int height, int w) {
    int row = blockIdx.x;
    int tid = threadIdx.x; // 0 <= tid < w

    extern __shared__ cuDoubleComplex s_data[];
    cuDoubleComplex* s_tmp = s_data + w;

    // 將該 row 的資料載入 shared memory
    s_data[tid] = data[row * w + tid];
    __syncthreads();

    // bit-reversal
    int r_idx = bit_reversed_indices[tid];
    s_tmp[r_idx] = s_data[tid];
    __syncthreads();

    s_data[tid] = s_tmp[tid];
    __syncthreads();

    int log2_n = (int)log2((double)w);
    for (int stage = 1; stage <= log2_n; stage++) {
        int m = 1 << stage;
        int m2 = m >> 1;

        // 每個 thread 對應到一個 pair_idx
        int pairs_total = w / 2;
        int pair_idx = tid;

        if (pair_idx < pairs_total) {
            int group = pair_idx / m2;
            int pair = pair_idx % m2;
            int pos = group * m + pair;

            double angle = -2.0 * M_PI * pair / m;
            cuDoubleComplex w_comp = make_cuDoubleComplex(cos(angle), sin(angle));
            cuDoubleComplex u = s_data[pos];
            cuDoubleComplex t = multiplyComplex(w_comp, s_data[pos + m2]);

            // 注意：這裡的寫回需要確保沒有別的 thread 同時寫相同位置，
            // 但在FFT的butterfly計算中每個 pair_idx 對應一對位置，
            // 不同 pair_idx 不會寫到同一個 element。
            // 所以可以直接寫回 s_data。
            if (tid == pair_idx) { // tid == pair_idx 這裡沒有意義，只是表達清晰，直接寫即可
                s_data[pos] = addComplex(u, t);
                s_data[pos + m2] = subtractComplex(u, t);
            }
        }
        __syncthreads();
    }

    // 將結果寫回 global memory
    data[row * w + tid] = s_data[tid];
}

__global__ void fft_rows_kernel_all(cuDoubleComplex* data, int width) {
    int tid = threadIdx.x;
    int rid = blockIdx.x;
    for (int stage = 1; stage <= (int)log2((float)width); ++stage) {
        int m = 1 << stage;
        int m2 = m >> 1;
        for (int id = tid; id < width / 2; id += blockDim.x) {
            int group = id / m2;
            int pair = id % m2;
            int pos = rid * width + group * m + pair;
            double angle = -2.0 * M_PI * pair / m;
            cuDoubleComplex w = make_cuDoubleComplex(cos(angle), sin(angle));
            cuDoubleComplex t = multiplyComplex(w, data[pos + m2]);
            cuDoubleComplex u = data[pos];

            data[pos] = addComplex(u, t);
            data[pos + m2] = subtractComplex(u, t);
        }
        __syncthreads();
    }
}


__global__ void transpose_kernel_tile(cuDoubleComplex* input, cuDoubleComplex* output, int height, int width) {
    
    __shared__ cuDoubleComplex tile[TILE][TILE + 1];

    int idx = blockIdx.x * TILE + threadIdx.x;
    int idy = blockIdx.y * TILE + threadIdx.y;

    tile[threadIdx.y][threadIdx.x] = input[(idy) * width + idx];
    tile[threadIdx.y + BLOCK][threadIdx.x] = input[(idy + BLOCK) * width + idx];

    __syncthreads();

    idx = blockIdx.y * TILE + threadIdx.x;
    idy = blockIdx.x * TILE + threadIdx.y;

    output[(idy) * height + idx] = tile[threadIdx.x][threadIdx.y];
    output[(idy + BLOCK) * height + idx] = tile[threadIdx.x][threadIdx.y + BLOCK];
}

void process_image_gpu(unsigned char* h_input, unsigned char* h_output, 
                      int height, int width, int channels) {
    unsigned char *d_input, *d_gray;
    cuDoubleComplex *d_fft_data, *d_transposed;
    unsigned char *d_output;
    double *d_max_val;
    int *d_indices;
    
    size_t input_size = height * width * channels * sizeof(unsigned char);
    size_t gray_size = height * width * sizeof(unsigned char);
    int w = power_of_two(width);
    int h = power_of_two(height);
    size_t complex_size = h * w * sizeof(cuDoubleComplex);
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_gray, gray_size);
    cudaMalloc(&d_fft_data, complex_size);
    cudaMalloc(&d_transposed, complex_size);
    cudaMalloc(&d_output, gray_size);
    cudaMalloc(&d_max_val, sizeof(double));
    cudaMalloc(&d_indices, max(w, h) * sizeof(int));
    
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(32, 32);
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y
    );
    
    // Convert to grayscale and initialize FFT data
    convert_to_gray_kernel<<<gridDim, blockDim>>>(d_input, d_gray, height, width, channels);
    init_fft_data<<<gridDim, blockDim>>>(d_gray, d_fft_data, height, width, w, h);
    
    int threadsPerBlock = 256;
    int numBlocks;
    
    // Process rows
    int* h_indices = (int*)malloc(max(w, h) * sizeof(int));
    generate_bit_reversal_indices(h_indices, w);
    cudaMemcpy(d_indices, h_indices, w * sizeof(int), cudaMemcpyHostToDevice);

    bit_reversal_reorder_kernel_all<<<dim3(ceil(w, T), T), T>>>(d_fft_data, d_transposed, d_indices, w, h);
    cudaMemcpy(d_fft_data, d_transposed, complex_size, cudaMemcpyDeviceToDevice);

    fft_rows_kernel_all<<<h, T>>>(d_fft_data, w);
   
    // Transpose matrix
    transpose_kernel_tile<<<dim3(w / TILE, (h / TILE)), dim3(TILE, BLOCK)>>>(d_fft_data, d_transposed, h, w);

    // Process columns (now as rows in transposed matrix)
    generate_bit_reversal_indices(h_indices, h);
    cudaMemcpy(d_indices, h_indices, h * sizeof(int), cudaMemcpyHostToDevice);
    
    bit_reversal_reorder_kernel_all<<<dim3(ceil(h, T), T), T>>>(d_transposed, d_fft_data, d_indices, h, w);
    cudaMemcpy(d_transposed, d_fft_data, complex_size, cudaMemcpyDeviceToDevice);

    fft_rows_kernel_all<<<w, T>>>(d_transposed, h);
    
    // Transpose back
    transpose_kernel_tile<<<dim3(ceil(h, TILE), ceil(w, TILE)), dim3(TILE, BLOCK)>>>(d_transposed, d_fft_data, w, h);

    // Center frequency and convert to image
    cudaMemset(d_max_val, 0, sizeof(double));
    center_frequency_kernel<<<gridDim, blockDim>>>(d_fft_data, h, w);
    compute_max_val_kernel<<<gridDim, blockDim>>>(d_fft_data, d_max_val, height, width, w);
    convert_to_image_kernel<<<gridDim, blockDim>>>(d_fft_data, d_output, d_max_val, 
                                                  height, width, w);
    
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
    
    // 開始計時
    clock_gettime(CLOCK_MONOTONIC, &start);

    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;

    // Read input
    read_png(argv[1], &src, &height, &width, &channels);
    dst = (unsigned char *) malloc(height * width * sizeof(unsigned char));

    // Process everything on GPU
    process_image_gpu(src, dst, height, width, channels);

    // Write output
    write_png(argv[2], dst, height, width, 1);

    // Clean up
    free(src);
    free(dst);

    // 結束計時
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    // 計算執行時間（轉換為毫秒）
    time_used = (end.tv_sec - start.tv_sec) * 1000.0 + 
                (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    printf("Total execution time: %.2f ms\n", time_used);
    return 0;
}
// fuse fft to fft_rows_kernel_all