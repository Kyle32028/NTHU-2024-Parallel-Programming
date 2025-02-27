#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <time.h>

#define M_PI 3.14159265358979323846

using namespace std;

typedef complex<double> ComplexD;
typedef vector<vector<ComplexD>> MatrixD;
typedef thrust::complex<double> ComplexGPU;

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

__global__ void bit_reversal_reorder_kernel(
    thrust::complex<double>* input, thrust::complex<double>* output, int* bit_reversed_indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int reversed_idx = bit_reversed_indices[idx];
        output[reversed_idx] = input[idx];  // 根據反轉後的索引重排數據
    }
}

void gpu_bit_reversal(thrust::complex<double>* data, int n) {
    // 1. 在 CPU 上生成位元反轉索引表
    int* bit_reversed_indices = (int*)malloc(n * sizeof(int));
    generate_bit_reversal_indices(bit_reversed_indices, n);

    // 2. 在 GPU 上分配記憶體
    thrust::complex<double> *d_input, *d_output;
    int* d_indices;

    cudaMalloc(&d_input, n * sizeof(thrust::complex<double>));
    cudaMalloc(&d_output, n * sizeof(thrust::complex<double>));
    cudaMalloc(&d_indices, n * sizeof(int));

    // 3. 拷貝數據和索引表到 GPU
    cudaMemcpy(d_input, data, n * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, bit_reversed_indices, n * sizeof(int), cudaMemcpyHostToDevice);

    // 4. 配置 CUDA 核心
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    bit_reversal_reorder_kernel<<<gridSize, blockSize>>>(d_input, d_output, d_indices, n);

    // 5. 將重排結果拷回 CPU
    cudaMemcpy(data, d_output, n * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);

    // 6. 釋放記憶體
    free(bit_reversed_indices);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices);
}

__global__ void fft_kernel(thrust::complex<double>* data, int n, int stage) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = 1 << stage;           // Size of subsequences
    int m2 = m >> 1;              // Half of m

    if (idx < n / 2) {
        int group = idx / m2;
        int pair = idx % m2;

        int pos = group * m + pair;

        thrust::complex<double> w = thrust::exp(thrust::complex<double>(0, -2.0 * M_PI * pair / m));

        thrust::complex<double> t = w * data[pos + m2];
        thrust::complex<double> u = data[pos];

        data[pos] = u + t;
        data[pos + m2] = u - t;
    }
}

void gpu_fft(thrust::complex<double>* d_data, int n) {
    int log_n = log2(n);

    for (int stage = 1; stage <= log_n; ++stage) {
        // int m = 1 << stage;
        int threadsPerBlock = 256;
        int numButterflies = n / 2;
        int numBlocks = (numButterflies + threadsPerBlock - 1) / threadsPerBlock;

        fft_kernel<<<numBlocks, threadsPerBlock>>>(d_data, n, stage);

        // Synchronize to ensure all threads have completed
        cudaDeviceSynchronize();
    }
}

// FFT function (FFT)
inline void fft(vector<ComplexD>& data) {
    int n = data.size();

    // Copy data to device
    thrust::complex<double>* d_data;
    cudaMalloc(&d_data, n * sizeof(thrust::complex<double>));
    cudaMemcpy(d_data, data.data(), n * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);

    // Perform bit reversal on GPU
    gpu_bit_reversal(d_data, n);

    // Perform FFT on GPU
    gpu_fft(d_data, n);

    // Copy result back to host
    cudaMemcpy(data.data(), d_data, n * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
}

// CUDA kernel for frequency centering
__global__ void center_frequency_kernel(ComplexGPU* data, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    int half_h = height / 2;
    int half_w = width / 2;
    
    // 只處理左上象限的點，對應的其他三個象限會自動完成交換
    if (idx < half_w && idy < half_h) {
        // 計算四個要交換的位置
        int pos1 = idy * width + idx;                          // 左上
        int pos2 = (idy + half_h) * width + (idx + half_w);   // 右下
        int pos3 = idy * width + (idx + half_w);              // 右上
        int pos4 = (idy + half_h) * width + idx;              // 左下
        
        // 交換數據
        ComplexGPU temp1 = data[pos1];
        ComplexGPU temp2 = data[pos3];
        
        data[pos1] = data[pos2];
        data[pos2] = temp1;
        data[pos3] = data[pos4];
        data[pos4] = temp2;
    }
}

void center_frequency(MatrixD& data, int height, int width) {
    // 分配 GPU 記憶體並複製數據
    ComplexGPU* d_data;
    size_t size = height * width * sizeof(ComplexGPU);
    
    // 將 2D vector 轉換為 1D array
    ComplexGPU* h_data = new ComplexGPU[height * width];
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            h_data[i * width + j] = ComplexGPU(data[i][j].real(), data[i][j].imag());
        }
    }
    
    // 分配 GPU 記憶體並複製數據
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // 設定 kernel 執行配置
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (width/2 + blockDim.x - 1) / blockDim.x,
        (height/2 + blockDim.y - 1) / blockDim.y
    );
    
    // 執行 kernel
    center_frequency_kernel<<<gridDim, blockDim>>>(d_data, height, width);
    
    // 複製結果回 host
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    
    // 將結果轉回 MatrixD 格式
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            data[i][j] = ComplexD(h_data[i * width + j].real(), h_data[i * width + j].imag());
        }
    }
    
    // 釋放記憶體
    delete[] h_data;
    cudaFree(d_data);
    
    // 錯誤檢查
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

// 修改 fft kernel 以處理特定的一行/列
__global__ void fft_kernel_row(thrust::complex<double>* data, int n, int stage, int row, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = 1 << stage;           // Size of subsequences
    int m2 = m >> 1;              // Half of m

    if (idx < n / 2) {
        int group = idx / m2;
        int pair = idx % m2;
        int pos = row * width + group * m + pair;

        thrust::complex<double> w = thrust::exp(thrust::complex<double>(0, -2.0 * M_PI * pair / m));
        thrust::complex<double> t = w * data[pos + m2];
        thrust::complex<double> u = data[pos];

        data[pos] = u + t;
        data[pos + m2] = u - t;
    }
}

__global__ void fft_kernel_col(thrust::complex<double>* data, int n, int stage, int col, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = 1 << stage;           // Size of subsequences
    int m2 = m >> 1;              // Half of m

    if (idx < n / 2) {
        int group = idx / m2;
        int pair = idx % m2;
        int pos = (group * m + pair) * width + col;

        thrust::complex<double> w = thrust::exp(thrust::complex<double>(0, -2.0 * M_PI * pair / m));
        thrust::complex<double> t = w * data[pos + m2 * width];
        thrust::complex<double> u = data[pos];

        data[pos] = u + t;
        data[pos + m2 * width] = u - t;
    }
}

// 修改 GPU 上的 bit reversal，使其能夠處理跨步存取
__global__ void bit_reversal_col_kernel(thrust::complex<double>* data, int* indices, int n, int col, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int reversed_idx = indices[idx];
        if (idx < reversed_idx) {
            thrust::complex<double> temp = data[idx * width + col];
            data[idx * width + col] = data[reversed_idx * width + col];
            data[reversed_idx * width + col] = temp;
        }
    }
}

// 用於計算最大值的 kernel
__global__ void compute_max_val_kernel(thrust::complex<double>* data, double* max_val, int height, int width, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        double val = log(1 + thrust::abs(data[idy * w + idx]));
        atomicMax((unsigned long long*)max_val, __double_as_longlong(val));
    }
}

// 用於將複數數據轉換為圖像數據的 kernel
__global__ void convert_to_image_kernel(thrust::complex<double>* data, unsigned char* output, 
                                      double* max_val, int height, int width, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        double val = log(1 + thrust::abs(data[idy * w + idx]));
        output[idy * width + idx] = static_cast<unsigned char>((val / *max_val) * 255);
    }
}

void fft_process_2d(unsigned char* s, unsigned char* t, int height, int width) {
    int w = power_of_two(width);
    int h = power_of_two(height);
    size_t complex_size = h * w * sizeof(thrust::complex<double>);
    
    // 在 GPU 上分配記憶體
    thrust::complex<double> *d_data;
    cudaMalloc(&d_data, complex_size);
    
    // 分配並初始化 bit reversal 索引表
    int* h_indices = (int*)malloc(max(w, h) * sizeof(int));
    int* d_indices;
    cudaMalloc(&d_indices, max(w, h) * sizeof(int));
    
    // 將輸入數據轉換為複數並複製到 GPU
    vector<thrust::complex<double>> h_data(h * w);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            h_data[i * w + j] = thrust::complex<double>((double)s[i * width + j], 0);
        }
        // 填充剩餘部分為 0
        for (int j = width; j < w; ++j) {
            h_data[i * w + j] = thrust::complex<double>(0, 0);
        }
    }
    // 填充剩餘行為 0
    for (int i = height; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            h_data[i * w + j] = thrust::complex<double>(0, 0);
        }
    }
    
    cudaMemcpy(d_data, h_data.data(), complex_size, cudaMemcpyHostToDevice);
    
    // 對每一行進行 FFT
    int threadsPerBlock = 256;
    int numBlocks = (w / 2 + threadsPerBlock - 1) / threadsPerBlock;
    
    // 先生成行的 bit reversal 索引
    generate_bit_reversal_indices(h_indices, w);
    cudaMemcpy(d_indices, h_indices, w * sizeof(int), cudaMemcpyHostToDevice);
    
    // 對每一行進行 FFT
    for (int i = 0; i < h; ++i) {
        // 先對該行進行 bit reversal
        thrust::complex<double>* row = d_data + i * w;
        gpu_bit_reversal(row, w);
        
        // 對該行執行 FFT
        for (int stage = 1; stage <= (int)log2(w); ++stage) {
            fft_kernel_row<<<numBlocks, threadsPerBlock>>>(d_data, w, stage, i, w);
            cudaDeviceSynchronize();
        }
    }
    
    // 生成列的 bit reversal 索引
    generate_bit_reversal_indices(h_indices, h);
    cudaMemcpy(d_indices, h_indices, h * sizeof(int), cudaMemcpyHostToDevice);
    
    // 對每一列進行 FFT
    numBlocks = (h + threadsPerBlock - 1) / threadsPerBlock;
    for (int j = 0; j < w; ++j) {
        // 對該列進行 bit reversal
        bit_reversal_col_kernel<<<numBlocks, threadsPerBlock>>>(d_data, d_indices, h, j, w);
        cudaDeviceSynchronize();
        
        // 對該列執行 FFT
        for (int stage = 1; stage <= (int)log2(h); ++stage) {
            fft_kernel_col<<<(h/2 + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
                d_data, h, stage, j, w);
            cudaDeviceSynchronize();
        }
    }
    
    // 中心化頻率
    dim3 blockDim(16, 16);
    dim3 gridDim((w + blockDim.x - 1) / blockDim.x,
                 (h + blockDim.y - 1) / blockDim.y);
    center_frequency_kernel<<<gridDim, blockDim>>>(d_data, h, w);
    cudaDeviceSynchronize();
    
    // 計算頻譜幅度並正規化
    cudaMemcpy(h_data.data(), d_data, complex_size, cudaMemcpyDeviceToHost);
    
    // 分配設備上的輸出圖像記憶體和最大值記憶體
    unsigned char* d_output;
    double* d_max_val;
    cudaMalloc(&d_output, height * width * sizeof(unsigned char));
    cudaMalloc(&d_max_val, sizeof(double));
    
    // 初始化最大值為 0
    cudaMemset(d_max_val, 0, sizeof(double));
    
    // 計算最大值
    compute_max_val_kernel<<<gridDim, blockDim>>>(d_data, d_max_val, height, width, w);
    cudaDeviceSynchronize();
    
    // 轉換為圖像數據
    convert_to_image_kernel<<<gridDim, blockDim>>>(d_data, d_output, d_max_val, height, width, w);
    cudaDeviceSynchronize();
    
    // 將結果複製回主機
    cudaMemcpy(t, d_output, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // 釋放記憶體
    cudaFree(d_output);
    cudaFree(d_max_val);
    cudaFree(d_data);
    cudaFree(d_indices);
    free(h_indices);
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
    struct timespec start, end;
    double time_used;
    
    // 開始計時
    clock_gettime(CLOCK_MONOTONIC, &start);

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

    // 結束計時
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    // 計算執行時間（轉換為毫秒）
    time_used = (end.tv_sec - start.tv_sec) * 1000.0 + 
                (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    printf("Total execution time: %.2f ms\n", time_used);
    return 0;
}