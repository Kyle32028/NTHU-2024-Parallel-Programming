#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define DEV_NO_0 0
#define DEV_NO_1 1
cudaDeviceProp prop;

const int INF = ((1 << 30) - 1);
const int BLOCK_SIZE = 32;     // threads per block dimension
const int THREAD_WORK = 2;     // work per thread dimension
const int BLOCK_WORK = BLOCK_SIZE * THREAD_WORK;  // total work per block dimension

// Global variables
int n, m;
int *Dist;

// Device constants
__constant__ int const_n;
__constant__ int const_B;

typedef struct {
    int device_id;
    int *dist;
    cudaStream_t stream;
    int start_block;  // 開始處理的 block row index (含)
    int end_block;    // 結束處理的 block row index (不含)
} GPUData;

__global__ void phase1_kernel(int* dist, int r, int B) {
    __shared__ int block[BLOCK_WORK][BLOCK_WORK + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int block_start = r * BLOCK_WORK;
    
    block[ty][tx] = dist[(block_start + ty) * const_n + (block_start + tx)];
    block[ty][tx + 32] = dist[(block_start + ty) * const_n + (block_start + tx + 32)];
    block[ty + 32][tx] = dist[(block_start + ty + 32) * const_n + (block_start + tx)];
    block[ty + 32][tx + 32] = dist[(block_start + ty + 32) * const_n + (block_start + tx + 32)];
    __syncthreads();

    for(int k = 0; k < BLOCK_WORK; k++) {
        block[ty][tx] = min(block[ty][tx], block[ty][k] + block[k][tx]);
        block[ty][tx + 32] = min(block[ty][tx + 32], block[ty][k] + block[k][tx + 32]);
        block[ty + 32][tx] = min(block[ty + 32][tx], block[ty + 32][k] + block[k][tx]);
        block[ty + 32][tx + 32] = min(block[ty + 32][tx + 32], block[ty + 32][k] + block[k][tx + 32]);
        __syncthreads();
    }

    dist[(block_start + ty) * const_n + (block_start + tx)] = block[ty][tx];
    dist[(block_start + ty) * const_n + (block_start + tx + 32)] = block[ty][tx + 32];
    dist[(block_start + ty + 32) * const_n + (block_start + tx)] = block[ty + 32][tx];
    dist[(block_start + ty + 32) * const_n + (block_start + tx + 32)] = block[ty + 32][tx + 32];
}

__global__ void phase2_kernel(int* dist, int r, int B) {
    const int target_block = blockIdx.y * BLOCK_WORK;
    
    if (target_block == r * BLOCK_WORK) {
        return;
    }
    
    __shared__ int pivot[BLOCK_WORK][BLOCK_WORK + 1];
    __shared__ int current[BLOCK_WORK][BLOCK_WORK + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int block_start = r * BLOCK_WORK;
    const int block_id = blockIdx.x;  // 0 for row, 1 for column
    
    pivot[ty][tx] = dist[(block_start + ty) * const_n + (block_start + tx)];
    pivot[ty][tx + 32] = dist[(block_start + ty) * const_n + (block_start + tx + 32)];
    pivot[ty + 32][tx] = dist[(block_start + ty + 32) * const_n + (block_start + tx)];
    pivot[ty + 32][tx + 32] = dist[(block_start + ty + 32) * const_n + (block_start + tx + 32)];
    
    const int row_idx = block_id == 0 ? target_block : block_start;
    const int col_idx = block_id == 0 ? block_start : target_block;
    
    current[ty][tx] = dist[(row_idx + ty) * const_n + (col_idx + tx)];
    current[ty][tx + 32] = dist[(row_idx + ty) * const_n + (col_idx + tx + 32)];
    current[ty + 32][tx] = dist[(row_idx + ty + 32) * const_n + (col_idx + tx)];
    current[ty + 32][tx + 32] = dist[(row_idx + ty + 32) * const_n + (col_idx + tx + 32)];
    __syncthreads();
    
    for(int k = 0; k < BLOCK_WORK; k++) {
        int src1 = block_id == 0 ? current[ty][k] : pivot[ty][k];
        int src2 = block_id == 0 ? current[ty + 32][k] : pivot[ty + 32][k];
        int tgt1 = block_id == 0 ? pivot[k][tx] : current[k][tx];
        int tgt2 = block_id == 0 ? pivot[k][tx + 32] : current[k][tx + 32];
        
        current[ty][tx] = min(current[ty][tx], src1 + tgt1);
        current[ty][tx + 32] = min(current[ty][tx + 32], src1 + tgt2);
        current[ty + 32][tx] = min(current[ty + 32][tx], src2 + tgt1);
        current[ty + 32][tx + 32] = min(current[ty + 32][tx + 32], src2 + tgt2);
        __syncthreads();
    }
    
    dist[(row_idx + ty) * const_n + (col_idx + tx)] = current[ty][tx];
    dist[(row_idx + ty) * const_n + (col_idx + tx + 32)] = current[ty][tx + 32];
    dist[(row_idx + ty + 32) * const_n + (col_idx + tx)] = current[ty + 32][tx];
    dist[(row_idx + ty + 32) * const_n + (col_idx + tx + 32)] = current[ty + 32][tx + 32];
}

__global__ void phase3_kernel(int* dist, int r, int B, int start_block, int end_block) {
    __shared__ int row_pivot[BLOCK_WORK][BLOCK_WORK];
    __shared__ int col_pivot[BLOCK_WORK][BLOCK_WORK];
    __shared__ int current[BLOCK_WORK][BLOCK_WORK];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int block_start = r * BLOCK_WORK;

    // blockIdx.y 對應的 block row 索引
    const int block_y = (start_block + blockIdx.y) * BLOCK_WORK;
    const int block_x = blockIdx.x * BLOCK_WORK;

    // Load data into shared memory
    row_pivot[ty][tx] = dist[(block_y + ty) * const_n + (block_start + tx)];
    row_pivot[ty][tx + 32] = dist[(block_y + ty) * const_n + (block_start + tx + 32)];
    row_pivot[ty + 32][tx] = dist[(block_y + ty + 32) * const_n + (block_start + tx)];
    row_pivot[ty + 32][tx + 32] = dist[(block_y + ty + 32) * const_n + (block_start + tx + 32)];
    
    col_pivot[ty][tx] = dist[(block_start + ty) * const_n + (block_x + tx)];
    col_pivot[ty][tx + 32] = dist[(block_start + ty) * const_n + (block_x + tx + 32)];
    col_pivot[ty + 32][tx] = dist[(block_start + ty + 32) * const_n + (block_x + tx)];
    col_pivot[ty + 32][tx + 32] = dist[(block_start + ty + 32) * const_n + (block_x + tx + 32)];
    
    current[ty][tx] = dist[(block_y + ty) * const_n + (block_x + tx)];
    current[ty][tx + 32] = dist[(block_y + ty) * const_n + (block_x + tx + 32)];
    current[ty + 32][tx] = dist[(block_y + ty + 32) * const_n + (block_x + tx)];
    current[ty + 32][tx + 32] = dist[(block_y + ty + 32) * const_n + (block_x + tx + 32)];
    __syncthreads();

    // Update current block
    for(int k = 0; k < BLOCK_WORK; k++) {
        current[ty][tx] = min(current[ty][tx], row_pivot[ty][k] + col_pivot[k][tx]);
        current[ty][tx + 32] = min(current[ty][tx + 32], row_pivot[ty][k] + col_pivot[k][tx + 32]);
        current[ty + 32][tx] = min(current[ty + 32][tx], row_pivot[ty + 32][k] + col_pivot[k][tx]);
        current[ty + 32][tx + 32] = min(current[ty + 32][tx + 32], row_pivot[ty + 32][k] + col_pivot[k][tx + 32]);
    }

    // Write back
    dist[(block_y + ty) * const_n + (block_x + tx)] = current[ty][tx];
    dist[(block_y + ty) * const_n + (block_x + tx + 32)] = current[ty][tx + 32];
    dist[(block_y + ty + 32) * const_n + (block_x + tx)] = current[ty + 32][tx];
    dist[(block_y + ty + 32) * const_n + (block_x + tx + 32)] = current[ty + 32][tx + 32];
}

void initGPUData(GPUData *gpu_data, int device_id, int padded_n) {
    gpu_data->device_id = device_id;
    
    // 設置GPU
    cudaSetDevice(device_id);
    
    // 為每個GPU分配完整矩陣
    size_t full_size = padded_n * padded_n * sizeof(int);
    cudaMalloc((void**)&(gpu_data->dist), full_size);
    
    // 創建CUDA stream
    cudaStreamCreate(&(gpu_data->stream));
}

void computeMultiGPU(int *Dist, int padded_n) {
    GPUData gpu_data[2];
    int n_round = padded_n / BLOCK_WORK; // number of block rows/cols

    // GPU0 負責前一半的 block rows
    gpu_data[0].start_block = 0;
    gpu_data[0].end_block = n_round / 2;

    // GPU1 負責後一半的 block rows
    gpu_data[1].start_block = n_round / 2;
    gpu_data[1].end_block = n_round;
    
    // 初始化 GPU
    for(int i = 0; i < 2; i++) {
        initGPUData(&gpu_data[i], i, padded_n);
        cudaSetDevice(i);
        cudaMemcpy(gpu_data[i].dist, 
                   Dist,
                   padded_n * padded_n * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    // 設定kernel啟動參數
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 phase2_grid(2, n_round);

    for(int r = 0; r < n_round; r++) {
        // 在本 round 開始前，先確定 pivot block row 屬於哪張 GPU
        int pivot_owner = (r < gpu_data[0].end_block) ? 0 : 1;
        int other_gpu = 1 - pivot_owner;
        
        int pivot_row = r * BLOCK_WORK;
        int row_count = BLOCK_WORK;
        size_t copy_size = row_count * padded_n * sizeof(int);

        // 若 pivot row 為 pivot_owner 所屬，則在 round 開始前將其傳給另一方
        // 以便接下來的計算使用
        if (copy_size > 0) {
            cudaSetDevice(gpu_data[pivot_owner].device_id);
            cudaMemcpyPeerAsync(gpu_data[other_gpu].dist + pivot_row * padded_n,
                                other_gpu,
                                gpu_data[pivot_owner].dist + pivot_row * padded_n,
                                pivot_owner,
                                copy_size,
                                gpu_data[pivot_owner].stream);
        }

        // 同步確保 pivot row 已經傳輸完成
        for (int i = 0; i < 2; i++) {
            cudaSetDevice(gpu_data[i].device_id);
            cudaStreamSynchronize(gpu_data[i].stream);
        }

        // Phase 1和Phase 2在兩個GPU上都執行
        for(int i = 0; i < 2; i++) {
            cudaSetDevice(gpu_data[i].device_id);
            
            phase1_kernel<<<1, block, 0, gpu_data[i].stream>>>(gpu_data[i].dist, r, BLOCK_WORK);
            phase2_kernel<<<phase2_grid, block, 0, gpu_data[i].stream>>>(gpu_data[i].dist, r, BLOCK_WORK);
        }

        // Phase 3：各自處理自己負責的 block rows
        for (int i = 0; i < 2; i++) {
            cudaSetDevice(gpu_data[i].device_id);

            int block_rows_for_gpu = gpu_data[i].end_block - gpu_data[i].start_block;
            dim3 phase3_grid(n_round, block_rows_for_gpu);

            phase3_kernel<<<phase3_grid, block, 0, gpu_data[i].stream>>>(
                gpu_data[i].dist,
                r,
                BLOCK_WORK,
                gpu_data[i].start_block,
                gpu_data[i].end_block
            );
        }

        // 同步
        for (int i = 0; i < 2; i++) {
            cudaSetDevice(gpu_data[i].device_id);
            cudaStreamSynchronize(gpu_data[i].stream);
        }
    }
    
    // 所有迭代完成後，GPU0 只對它負責的那一半(row)是正確的，GPU1 只對它負責的那一半正確。
    // 因此需要將兩邊的部分合併到 host 上。
    {
        // 先把 GPU0 負責的部分拷回 host
        int gpu0_row_start = gpu_data[0].start_block * BLOCK_WORK; // 通常是0
        int gpu0_row_end = gpu_data[0].end_block * BLOCK_WORK;     // 這是 GPU0 負責的最後一行(不含)
        int gpu0_rows = gpu0_row_end - gpu0_row_start;             // GPU0 負責的行數

        cudaSetDevice(0);
        cudaMemcpy(Dist + gpu0_row_start * padded_n,
                   gpu_data[0].dist + gpu0_row_start * padded_n,
                   gpu0_rows * padded_n * sizeof(int),
                   cudaMemcpyDeviceToHost);

        // 把 GPU1 負責的部分拷回 host
        int gpu1_row_start = gpu_data[1].start_block * BLOCK_WORK;
        int gpu1_row_end = gpu_data[1].end_block * BLOCK_WORK;
        int gpu1_rows = gpu1_row_end - gpu1_row_start;

        cudaSetDevice(1);
        cudaMemcpy(Dist + gpu1_row_start * padded_n,
                   gpu_data[1].dist + gpu1_row_start * padded_n,
                   gpu1_rows * padded_n * sizeof(int),
                   cudaMemcpyDeviceToHost);
    }

    // 清理資源
    for(int i = 0; i < 2; i++) {
        cudaSetDevice(gpu_data[i].device_id);
        cudaStreamDestroy(gpu_data[i].stream);
        cudaFree(gpu_data[i].dist);
    }
}

void input(char* infile, int* padded_n_ptr) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // Pad n to a multiple of BLOCK_WORK (64)
    int padded_n = ((n + BLOCK_WORK - 1) / BLOCK_WORK) * BLOCK_WORK;
    *padded_n_ptr = padded_n;

    cudaHostAlloc((void**)&Dist, padded_n * padded_n * sizeof(int), cudaHostAllocDefault);

    // Initialize the padded distance matrix
    for (int i = 0; i < padded_n; ++i) {
        for (int j = 0; j < padded_n; ++j) {
            if (i == j) {
                Dist[i * padded_n + j] = 0;
            } else {
                Dist[i * padded_n + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * padded_n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName, int padded_n) {
    FILE* outfile = fopen(outFileName, "wb");
    for (int i = 0; i < n; ++i) {
        fwrite(&Dist[i * padded_n], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int main(int argc, char* argv[]) {
    int padded_n;
    input(argv[1], &padded_n);

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);

    // 設置常數記憶體
    for(int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaMemcpyToSymbol(const_n, &padded_n, sizeof(int));
        cudaMemcpyToSymbol(const_B, &BLOCK_WORK, sizeof(int));
    }

    // 調用多GPU計算函數
    computeMultiGPU(Dist, padded_n);

    output(argv[2], padded_n);
    cudaFreeHost(Dist);
    
    return 0;
}