#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_FILTER_NONE);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

// Helper function for non-SIMD calculation of remaining points
void mandelbrot_calculate_point(double x0, double y0, int max_iters, int* result) {
    double x = 0, y = 0;
    int repeats = 0;
    while (repeats < max_iters && x*x + y*y < 4) {
        double temp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = temp;
        repeats++;
    }
    *result = repeats;
}

// SIMD-accelerated function to compute the Mandelbrot set for multiple rows
void compute_mandelbrot_rows(int* rows_data, int* row_indices, int num_rows,
                           int width, int height, int iters,
                           double left, double right, double lower, double upper) {
    double x0_step = (right - left) / width;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < num_rows; i++) {
        int row = row_indices[i];
        double y0 = row * ((upper - lower) / height) + lower;
        
        // SIMD variables
        __m512d v_four = _mm512_set1_pd(4.0);
        __m512d v_two = _mm512_set1_pd(2.0);
        __m512d v_x0, v_y0, v_x, v_y, v_xx, v_yy, v_xy, v_length_squared;
        __m512i v_repeats, v_one, v_iters;
        __mmask8 mask;

        v_one = _mm512_set1_epi64(1);
        v_iters = _mm512_set1_epi64(iters);
        v_y0 = _mm512_set1_pd(y0);

        // Process 8 pixels at a time using AVX-512
        int x;
        for (x = 0; x + 7 < width; x += 8) {
            v_x0 = _mm512_set_pd(
                (x + 7) * x0_step + left,
                (x + 6) * x0_step + left,
                (x + 5) * x0_step + left,
                (x + 4) * x0_step + left,
                (x + 3) * x0_step + left,
                (x + 2) * x0_step + left,
                (x + 1) * x0_step + left,
                x * x0_step + left
            );

            v_x = _mm512_setzero_pd();
            v_y = _mm512_setzero_pd();
            v_repeats = _mm512_setzero_si512();

            mask = 0xFF;  // All 8 bits set
            for (int k = 0; k < iters && mask; ++k) {
                // Compute x and y squares
                v_xx = _mm512_mul_pd(v_x, v_x);
                v_yy = _mm512_mul_pd(v_y, v_y);

                v_length_squared = _mm512_add_pd(v_xx, v_yy);
                mask = _mm512_cmp_pd_mask(v_length_squared, v_four, _CMP_LT_OQ);
                
                // Update x and y
                v_xy = _mm512_mul_pd(v_x, v_y);
                v_x = _mm512_add_pd(_mm512_sub_pd(v_xx, v_yy), v_x0);
                v_y = _mm512_fmadd_pd(v_two, v_xy, v_y0);

                v_repeats = _mm512_mask_add_epi64(v_repeats, mask, v_repeats, v_one);
            }

            int64_t repeats[8];
            _mm512_storeu_si512((__m512i*)repeats, v_repeats);

            for (int k = 0; k < 8; ++k) {
                rows_data[i * width + x + k] = (int)repeats[k];
            }
        }

        // Handle remaining pixels sequentially
        for (; x < width; ++x) {
            double x0 = x * x0_step + left;
            mandelbrot_calculate_point(x0, y0, iters, &rows_data[i * width + x]);
        }
    }
}

int main(int argc, char** argv) {
    int rank, num_procs;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Parse arguments
    // assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);
    
    // Set number of OpenMP threads based on available CPUs
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int num_cpus = CPU_COUNT(&cpu_set);
    omp_set_num_threads(num_cpus);
    
    // Calculate exact number of rows for this process
    int my_row_count = 0;
    for (int row = rank; row < height; row += num_procs) {
        my_row_count++;
    }
    
    // Allocate arrays for row indices and data
    int* row_indices = (int*)malloc(my_row_count * sizeof(int));
    int* local_rows_data = (int*)malloc(my_row_count * width * sizeof(int));
    assert(row_indices && local_rows_data);
    
    // Calculate row indices for this process
    int idx = 0;
    for (int row = rank; row < height; row += num_procs) {
        row_indices[idx++] = row;
    }
    
    // Compute assigned rows using SIMD-accelerated function
    compute_mandelbrot_rows(local_rows_data, row_indices, my_row_count,
                          width, height, iters, left, right, lower, upper);
    
    // Gather the row counts from all processes
    int* all_row_counts = NULL;
    if (rank == 0) {
        all_row_counts = (int*)malloc(num_procs * sizeof(int));
        assert(all_row_counts);
    }
    
    // Gather number of rows from each process
    MPI_Gather(&my_row_count, 1, MPI_INT, all_row_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Prepare arrays for gathering data
    int* displs = NULL;
    int* row_displs = NULL;
    int* all_row_indices = NULL;
    int* full_image = NULL;
    
    if (rank == 0) {
        displs = (int*)malloc(num_procs * sizeof(int));
        row_displs = (int*)malloc(num_procs * sizeof(int));
        all_row_indices = (int*)malloc(height * sizeof(int));
        full_image = (int*)malloc(width * height * sizeof(int));
        assert(displs && row_displs && all_row_indices && full_image);
        
        // Calculate displacements
        displs[0] = 0;
        row_displs[0] = 0;
        for (int i = 1; i < num_procs; i++) {
            displs[i] = displs[i-1] + all_row_counts[i-1];
            row_displs[i] = displs[i];
        }
    }
    
    // Gather row indices
    MPI_Gatherv(row_indices, my_row_count, MPI_INT,
                all_row_indices, all_row_counts, row_displs,
                MPI_INT, 0, MPI_COMM_WORLD);
    
    // Prepare counts for data gathering (multiply by width)
    int my_data_count = my_row_count * width;
    int* all_data_counts = NULL;
    int* data_displs = NULL;
    
    if (rank == 0) {
        all_data_counts = (int*)malloc(num_procs * sizeof(int));
        data_displs = (int*)malloc(num_procs * sizeof(int));
        assert(all_data_counts && data_displs);
        
        for (int i = 0; i < num_procs; i++) {
            all_data_counts[i] = all_row_counts[i] * width;
            data_displs[i] = displs[i] * width;
        }
    }
    
    // Gather computed data
    MPI_Gatherv(local_rows_data, my_data_count, MPI_INT,
                full_image, all_data_counts, data_displs,
                MPI_INT, 0, MPI_COMM_WORLD);
    
    // Rank 0 reorganizes the data into the final image
    if (rank == 0) {
        int* temp_image = (int*)malloc(width * height * sizeof(int));
        assert(temp_image);
        
        // Copy data to correct positions
        for (int i = 0; i < num_procs; i++) {
            for (int j = 0; j < all_row_counts[i]; j++) {
                int src_row = j;
                int dst_row = all_row_indices[displs[i] + j];
                memcpy(temp_image + dst_row * width,
                      full_image + (data_displs[i] + j * width),
                      width * sizeof(int));
            }
        }
        
        // Write the final image
        write_png(filename, iters, width, height, temp_image);
        
        // Cleanup
        free(temp_image);
        free(all_row_counts);
        free(all_data_counts);
        free(data_displs);
        free(displs);
        free(row_displs);
        free(all_row_indices);
        free(full_image);
    }
    
    // Cleanup
    free(row_indices);
    free(local_rows_data);
    
    MPI_Finalize();
    return 0;
}