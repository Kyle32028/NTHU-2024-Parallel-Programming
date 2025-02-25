#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BR 64
#define BC 64

int B, N, d;
float *Q, *K, *V, *O;
float *d_Q, *d_K, *d_V, *d_O;
float *d_m, *d_l;

double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

__device__ inline float _maxf_custom(float a, float b) { 
    return a > b ? a : b; 
}

// warp 級別的 reduce-max
__device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// warp 級別的 reduce-sum
__device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val += other;
    }
    return val;
}

__global__ void FlashAttentionKernel(
    const float * __restrict__ Q, const float * __restrict__ K, const float * __restrict__ V,
    float *O, float *m, float *l,
    int B, int N, int d, int tr, int tc, float inv_sqrt_d)
{
    __shared__ float Q_sh[BR][64 + 1];
    __shared__ float K_sh[BC][64 + 1];

    int br = BR, bc = BC;
    int tcol = threadIdx.x; 
    int trow = threadIdx.y; 

    // blockIdx.x 封裝了哪個 batch 與哪個 i-block
    int global_block_id = blockIdx.x;
    int b_idx = global_block_id / tr; 
    int i_idx = global_block_id % tr; 

    int base_q = b_idx * (N * d) + i_idx * br * d;  
    int base_o = b_idx * (N * d) + i_idx * br * d;  
    int base_m = b_idx * N + i_idx * br;           
    int base_l = b_idx * N + i_idx * br;           
    int base_kv = b_idx * (N * d);

    // 這兩個 row 代表該 thread 負責的 row index（2D blockDim 情況）
    int row0 = trow * 2;
    int row1 = trow * 2 + 1;

    //-----------------------------------------
    // 1. 載入當前 row-block 對應的 m, l（放到暫存變數中）
    //    注意：row1 可能會超過 BR-1（若 BR=64, trow=31 時 row1=63 還合法）
    //    若擔心越界需先判斷，但此處假設 (BR=64, blockDim=(32,32)) 足夠覆蓋
    //-----------------------------------------
    float mi0 = -FLT_MAX;
    float li0 = 0.0f;
    float mi1 = -FLT_MAX;
    float li1 = 0.0f;

    //-----------------------------------------
    // 2. 載入 Q block 至 shared memory: (BR x d)
    //-----------------------------------------
    for (int q_r = trow; q_r < br; q_r += blockDim.y) {
        for (int q_c = tcol; q_c < d; q_c += blockDim.x) {
            Q_sh[q_r][q_c] = Q[base_q + q_r*d + q_c];
        }
    }
    __syncthreads();

    //-----------------------------------------
    // 3. 在 kernel 裡面迴圈處理所有的 j-block (tc = N/bc)
    //-----------------------------------------
    for (int j = 0; j < tc; j++) {
        int base_col = j * bc;

        //-------------------------------------
        // 3.1 載入 K block 至 shared memory (BC x d)
        //-------------------------------------
        for (int k_r = trow; k_r < bc; k_r += blockDim.y) {
            int k_idx = base_col + k_r;
            for (int k_c = tcol; k_c < d; k_c += blockDim.x) {
                K_sh[k_r][k_c] = K[base_kv + k_idx*d + k_c];
            }
        }
        __syncthreads();

        //-------------------------------------
        // 3.2 計算 QK^T * inv_sqrt_d（每個 thread 負責 2x2 tile）
        //-------------------------------------
        int col0 = tcol * 2;
        int col1 = tcol * 2 + 1;

        // 若 row0, row1, col0, col1 超過範圍，需判斷，但此處示範時略
        float sum_val_00 = 0.0f;
        float sum_val_01 = 0.0f;
        float sum_val_10 = 0.0f;
        float sum_val_11 = 0.0f;

        #pragma unroll 4
        for (int t = 0; t < d; t++) {
            float q_val_0 = Q_sh[row0][t];
            float q_val_1 = Q_sh[row1][t];
            float k_val_0 = K_sh[col0][t];
            float k_val_1 = K_sh[col1][t];

            sum_val_00 += q_val_0 * k_val_0;
            sum_val_01 += q_val_0 * k_val_1;
            sum_val_10 += q_val_1 * k_val_0;
            sum_val_11 += q_val_1 * k_val_1;
        }

        float val_sij_00 = sum_val_00 * inv_sqrt_d;
        float val_sij_01 = sum_val_01 * inv_sqrt_d;
        float val_sij_10 = sum_val_10 * inv_sqrt_d;
        float val_sij_11 = sum_val_11 * inv_sqrt_d;

        //-------------------------------------
        // 3.3 reduce-max per row (warp 級)
        //-------------------------------------
        float row_val0 = fmaxf(val_sij_00, val_sij_01);
        float row_max0 = warpReduceMax(row_val0);

        float row_val1 = fmaxf(val_sij_10, val_sij_11);
        float row_max1 = warpReduceMax(row_val1);

        // 需要將 row_max0/row_max1 廣播給該warp內所有 threads
        // 可以用 __shfl_sync 取出 lane 0 的值
        float warp_max0 = __shfl_sync(0xffffffff, row_max0, 0);
        float warp_max1 = __shfl_sync(0xffffffff, row_max1, 0);

        //-------------------------------------
        // 3.4 計算 exp(sij - max)
        //-------------------------------------
        float val_pij_00 = expf(val_sij_00 - warp_max0);
        float val_pij_01 = expf(val_sij_01 - warp_max0);
        float val_pij_10 = expf(val_sij_10 - warp_max1);
        float val_pij_11 = expf(val_sij_11 - warp_max1);

        // 再做一次 warp 級的 reduce-sum
        float row_sum0_thread = val_pij_00 + val_pij_01;
        float row_sum0 = warpReduceSum(row_sum0_thread);

        float row_sum1_thread = val_pij_10 + val_pij_11;
        float row_sum1 = warpReduceSum(row_sum1_thread);

        // 需要將 row_sum0/row_sum1 廣播給warp所有 threads
        float warp_sum0 = __shfl_sync(0xffffffff, row_sum0, 0);
        float warp_sum1 = __shfl_sync(0xffffffff, row_sum1, 0);

        //-------------------------------------
        // 3.5 更新 mi, li
        //
        //  mi_new = max(mi_old, warp_max)
        //  li_new = e^(mi_old - mi_new)*li_old + e^(warp_max - mi_new)*warp_sum
        //-------------------------------------
        float mi_new0 = _maxf_custom(mi0, warp_max0);
        float li_new0 = expf(mi0 - mi_new0)*li0 + expf(warp_max0 - mi_new0)*warp_sum0;

        float mi_new1 = _maxf_custom(mi1, warp_max1);
        float li_new1 = expf(mi1 - mi_new1)*li1 + expf(warp_max1 - mi_new1)*warp_sum1;

        //-------------------------------------
        // 3.6 更新 O (類似一次性融合)
        //   O_new = [ e^(mi_old - mi_new)*li_old*O_old + e^(warp_max - mi_new)* Σ p_ij * V_j ] / li_new
        //-------------------------------------
        // 對 d 個維度逐一做加權更新
        for (int dim_idx = 0; dim_idx < d; dim_idx++) {
            // 讀取該維度上對應的 V
            float v_val_00 = V[base_kv + (base_col + col0)*d + dim_idx];
            float v_val_01 = V[base_kv + (base_col + col1)*d + dim_idx];
            float v_val_10 = v_val_00; // 同 base_col+col0
            float v_val_11 = v_val_01; // 同 base_col+col1

            float pv_val_00 = val_pij_00 * v_val_00;
            float pv_val_01 = val_pij_01 * v_val_01;
            float pv_val_10 = val_pij_10 * v_val_10;
            float pv_val_11 = val_pij_11 * v_val_11;

            // 先 warp reduce sum(2x2)
            float pv_sum0 = warpReduceSum(pv_val_00 + pv_val_01);
            float pv_sum1 = warpReduceSum(pv_val_10 + pv_val_11);

            // 只用 tcol==0 的 thread 寫回即可，避免重複寫
            if (tcol == 0) {
                // 讀取 O_old
                float old_o0 = O[base_o + row0*d + dim_idx];
                float old_o1 = O[base_o + row1*d + dim_idx];

                // numerator = e^(mi_old - mi_new)*li_old*O_old + e^(warp_max - mi_new)* Σ p_ij V_j
                float numerator0 = expf(mi0 - mi_new0)*li0*old_o0 + expf(warp_max0 - mi_new0)*pv_sum0;
                float numerator1 = expf(mi1 - mi_new1)*li1*old_o1 + expf(warp_max1 - mi_new1)*pv_sum1;

                // O_new = numerator / li_new
                O[base_o + row0*d + dim_idx] = numerator0 / li_new0;
                O[base_o + row1*d + dim_idx] = numerator1 / li_new1;
            }
        }

        // 更新當前 thread row 對應的 mi0, li0, mi1, li1，以便下一次 j-block
        mi0 = mi_new0;
        li0 = li_new0;
        mi1 = mi_new1;
        li1 = li_new1;

        __syncthreads();
    }
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");
    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));
    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");
    fwrite(O, sizeof(float), B * N * d, file);
    fclose(file);
}

int main(int argc, char *argv[]) {
    input(argv[1]);

    // double start, end;
    // start = getTimeStamp();

    cudaMalloc((void**)&d_Q, B*N*d*sizeof(float));
    cudaMalloc((void**)&d_K, B*N*d*sizeof(float));
    cudaMalloc((void**)&d_V, B*N*d*sizeof(float));
    cudaMalloc((void**)&d_O, B*N*d*sizeof(float));
    cudaMalloc((void**)&d_m, B*N*sizeof(float));
    cudaMalloc((void**)&d_l, B*N*sizeof(float));

    cudaMemcpy(d_Q, Q, B*N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, B*N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, B*N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, O, B*N*d*sizeof(float), cudaMemcpyHostToDevice);

    // 初始化 m, l
    {
        float *m_init = (float*)malloc(B*N*sizeof(float));
        float *l_init = (float*)malloc(B*N*sizeof(float));
        for (int i = 0; i < B*N; i++) {
            m_init[i] = -FLT_MAX;
            l_init[i] = 0.0f;
        }
        cudaMemcpy(d_m, m_init, B*N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_l, l_init, B*N*sizeof(float), cudaMemcpyHostToDevice);
        free(m_init);
        free(l_init);
    }

    int br = BR, bc = BC;
    int tr = N/br; 
    int tc = N/bc;

    float inv_sqrt_d = 1.0f / sqrtf((float)d);

    dim3 blockDim(32, 32);
    dim3 gridDim(B*tr, 1);

    FlashAttentionKernel<<<gridDim, blockDim>>>(
        d_Q, d_K, d_V, d_O, d_m, d_l, B, N, d, tr, tc, inv_sqrt_d
    );
    cudaDeviceSynchronize();

    cudaMemcpy(O, d_O, B*N*d*sizeof(float), cudaMemcpyDeviceToHost);

    // end = getTimeStamp();
    // printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    // printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_m);
    cudaFree(d_l);

    free(Q);
    free(K);
    free(V);
    free(O);

    return 0;
}
