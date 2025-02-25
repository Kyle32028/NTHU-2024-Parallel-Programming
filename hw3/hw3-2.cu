#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

// 可透過修改BLOCK_SIZE測試不同的block大小 (必須是2的次方並可整除64較佳)
#define BLOCK_SIZE 32
#define THREAD_WORK 2
#define BLOCK_WORK (BLOCK_SIZE * THREAD_WORK)

#define DEV_NO 0
cudaDeviceProp prop;

const int INF = ((1 << 30) - 1);

// 全域變數
int n, m;
int *Dist;

// Device constants
__constant__ int const_n;
__constant__ int const_B;

__global__ void phase1_kernel(int* dist, int r, int B) {
    __shared__ int block[BLOCK_WORK][BLOCK_WORK + 1]; 
    // 使用 +1 是為了記憶體對齊，保持原先的設計
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int block_start = r * BLOCK_WORK;

    int half_work = BLOCK_WORK / 2;
    
    // 載入block，此處原本+32的地方改為+half_work
    block[ty][tx] = dist[(block_start + ty) * const_n + (block_start + tx)];
    block[ty][tx + half_work] = dist[(block_start + ty) * const_n + (block_start + tx + half_work)];
    block[ty + half_work][tx] = dist[(block_start + ty + half_work) * const_n + (block_start + tx)];
    block[ty + half_work][tx + half_work] = dist[(block_start + ty + half_work) * const_n + (block_start + tx + half_work)];
    __syncthreads();

    // Floyd-Warshall block內部計算
    for (int k = 0; k < BLOCK_WORK; k++) {
        block[ty][tx] = min(block[ty][tx], block[ty][k] + block[k][tx]);
        block[ty][tx + half_work] = min(block[ty][tx + half_work], block[ty][k] + block[k][tx + half_work]);
        block[ty + half_work][tx] = min(block[ty + half_work][tx], block[ty + half_work][k] + block[k][tx]);
        block[ty + half_work][tx + half_work] = min(block[ty + half_work][tx + half_work], block[ty + half_work][k] + block[k][tx + half_work]);
        __syncthreads();
    }

    // 寫回global memory
    dist[(block_start + ty) * const_n + (block_start + tx)] = block[ty][tx];
    dist[(block_start + ty) * const_n + (block_start + tx + half_work)] = block[ty][tx + half_work];
    dist[(block_start + ty + half_work) * const_n + (block_start + tx)] = block[ty + half_work][tx];
    dist[(block_start + ty + half_work) * const_n + (block_start + tx + half_work)] = block[ty + half_work][tx + half_work];
}

__global__ void phase2_kernel(int* dist, int r, int B) {
    const int target_block = blockIdx.y * BLOCK_WORK;
    if (target_block == r * BLOCK_WORK) return;

    __shared__ int pivot[BLOCK_WORK][BLOCK_WORK + 1];
    __shared__ int current[BLOCK_WORK][BLOCK_WORK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int block_start = r * BLOCK_WORK;
    const int block_id = blockIdx.x;  // 0 for row, 1 for column
    int half_work = BLOCK_WORK / 2;

    // 載入 pivot block
    pivot[ty][tx] = dist[(block_start + ty) * const_n + (block_start + tx)];
    pivot[ty][tx + half_work] = dist[(block_start + ty) * const_n + (block_start + tx + half_work)];
    pivot[ty + half_work][tx] = dist[(block_start + ty + half_work) * const_n + (block_start + tx)];
    pivot[ty + half_work][tx + half_work] = dist[(block_start + ty + half_work) * const_n + (block_start + tx + half_work)];

    const int row_idx = (block_id == 0) ? target_block : block_start;
    const int col_idx = (block_id == 0) ? block_start : target_block;

    // 載入 current block
    current[ty][tx] = dist[(row_idx + ty) * const_n + (col_idx + tx)];
    current[ty][tx + half_work] = dist[(row_idx + ty) * const_n + (col_idx + tx + half_work)];
    current[ty + half_work][tx] = dist[(row_idx + ty + half_work) * const_n + (col_idx + tx)];
    current[ty + half_work][tx + half_work] = dist[(row_idx + ty + half_work) * const_n + (col_idx + tx + half_work)];
    __syncthreads();

    // 更新 current block
    for (int k = 0; k < BLOCK_WORK; k++) {
        int src1 = (block_id == 0) ? current[ty][k] : pivot[ty][k];
        int src2 = (block_id == 0) ? current[ty + half_work][k] : pivot[ty + half_work][k];
        int tgt1 = (block_id == 0) ? pivot[k][tx] : current[k][tx];
        int tgt2 = (block_id == 0) ? pivot[k][tx + half_work] : current[k][tx + half_work];

        current[ty][tx] = min(current[ty][tx], src1 + tgt1);
        current[ty][tx + half_work] = min(current[ty][tx + half_work], src1 + tgt2);
        current[ty + half_work][tx] = min(current[ty + half_work][tx], src2 + tgt1);
        current[ty + half_work][tx + half_work] = min(current[ty + half_work][tx + half_work], src2 + tgt2);
        __syncthreads();
    }

    // 寫回
    dist[(row_idx + ty) * const_n + (col_idx + tx)] = current[ty][tx];
    dist[(row_idx + ty) * const_n + (col_idx + tx + half_work)] = current[ty][tx + half_work];
    dist[(row_idx + ty + half_work) * const_n + (col_idx + tx)] = current[ty + half_work][tx];
    dist[(row_idx + ty + half_work) * const_n + (col_idx + tx + half_work)] = current[ty + half_work][tx + half_work];
}

__global__ void phase3_kernel(int* dist, int r, int B) {
    __shared__ int row_pivot[BLOCK_WORK][BLOCK_WORK];
    __shared__ int col_pivot[BLOCK_WORK][BLOCK_WORK];
    __shared__ int current[BLOCK_WORK][BLOCK_WORK];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int block_start = r * BLOCK_WORK;
    const int block_y = blockIdx.y * BLOCK_WORK;
    const int block_x = blockIdx.x * BLOCK_WORK;
    int half_work = BLOCK_WORK / 2;

    // 載入 row_pivot
    row_pivot[ty][tx] = dist[(block_y + ty)*const_n + (block_start + tx)];
    row_pivot[ty][tx + half_work] = dist[(block_y + ty)*const_n + (block_start + tx + half_work)];
    row_pivot[ty + half_work][tx] = dist[(block_y + ty + half_work)*const_n + (block_start + tx)];
    row_pivot[ty + half_work][tx + half_work] = dist[(block_y + ty + half_work)*const_n + (block_start + tx + half_work)];

    // 載入 col_pivot
    col_pivot[ty][tx] = dist[(block_start + ty)*const_n + (block_x + tx)];
    col_pivot[ty][tx + half_work] = dist[(block_start + ty)*const_n + (block_x + tx + half_work)];
    col_pivot[ty + half_work][tx] = dist[(block_start + ty + half_work)*const_n + (block_x + tx)];
    col_pivot[ty + half_work][tx + half_work] = dist[(block_start + ty + half_work)*const_n + (block_x + tx + half_work)];

    // 載入 current
    current[ty][tx] = dist[(block_y + ty)*const_n + (block_x + tx)];
    current[ty][tx + half_work] = dist[(block_y + ty)*const_n + (block_x + tx + half_work)];
    current[ty + half_work][tx] = dist[(block_y + ty + half_work)*const_n + (block_x + tx)];
    current[ty + half_work][tx + half_work] = dist[(block_y + ty + half_work)*const_n + (block_x + tx + half_work)];
    __syncthreads();

    // 更新 current block
    for (int k = 0; k < BLOCK_WORK; k++) {
        current[ty][tx] = min(current[ty][tx], row_pivot[ty][k] + col_pivot[k][tx]);
        current[ty][tx + half_work] = min(current[ty][tx + half_work], row_pivot[ty][k] + col_pivot[k][tx + half_work]);
        current[ty + half_work][tx] = min(current[ty + half_work][tx], row_pivot[ty + half_work][k] + col_pivot[k][tx]);
        current[ty + half_work][tx + half_work] = min(current[ty + half_work][tx + half_work], row_pivot[ty + half_work][k] + col_pivot[k][tx + half_work]);
    }

    // 寫回
    dist[(block_y + ty)*const_n + (block_x + tx)] = current[ty][tx];
    dist[(block_y + ty)*const_n + (block_x + tx + half_work)] = current[ty][tx + half_work];
    dist[(block_y + ty + half_work)*const_n + (block_x + tx)] = current[ty + half_work][tx];
    dist[(block_y + ty + half_work)*const_n + (block_x + tx + half_work)] = current[ty + half_work][tx + half_work];
}

void input(char* infile, int* padded_n_ptr) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // Pad n to a multiple of BLOCK_WORK (64)
    int padded_n = ((n + BLOCK_WORK - 1) / BLOCK_WORK) * BLOCK_WORK;
    *padded_n_ptr = padded_n;

    cudaHostAlloc((void**)&Dist, padded_n * padded_n * sizeof(int), cudaHostAllocDefault);

    // 使用OpenMP平行化初始化 Dist 矩陣
    #pragma omp parallel for collapse(2) schedule(static)
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

    int *dist_gpu;
    size_t dist_size = padded_n * padded_n * sizeof(int);
    cudaMalloc((void**)&dist_gpu, dist_size);
    cudaMemcpy(dist_gpu, Dist, dist_size, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(const_n, &padded_n, sizeof(int));

    int block_work = BLOCK_SIZE * THREAD_WORK;
    cudaMemcpyToSymbol(const_B, &block_work, sizeof(int));

    int n_round = padded_n / BLOCK_WORK;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 phase2_grid(2, n_round);
    dim3 phase3_grid(n_round, n_round);

    for (int r = 0; r < n_round; r++) {
        phase1_kernel<<<1, block>>>(dist_gpu, r, BLOCK_WORK);
        phase2_kernel<<<phase2_grid, block>>>(dist_gpu, r, BLOCK_WORK);
        phase3_kernel<<<phase3_grid, block>>>(dist_gpu, r, BLOCK_WORK);
    }

    cudaMemcpy(Dist, dist_gpu, dist_size, cudaMemcpyDeviceToHost);
    output(argv[2], padded_n);

    cudaFree(dist_gpu);
    cudaFreeHost(Dist);
    return 0;
}