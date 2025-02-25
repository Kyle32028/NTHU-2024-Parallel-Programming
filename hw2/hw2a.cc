#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>

typedef struct {
    int id;
    int thread_count;
    int width;
    int height;
    int iters;
    double left;
    double right;
    double lower;
    double upper;
    int* image;
} thread_arg_t;

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

void* mandelbrot_thread(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    
    __m512d v_four = _mm512_set1_pd(4.0);
    __m512d v_two = _mm512_set1_pd(2.0);
    __m512d v_x0, v_y0, v_x, v_y, v_xx, v_yy, v_xy, v_length_squared;
    __m512i v_repeats, v_one, v_iters;
    __mmask8 mask;

    v_one = _mm512_set1_epi64(1);
    v_iters = _mm512_set1_epi64(targ->iters);

    double x0_step = (targ->right - targ->left) / targ->width;
    
    for (int j = targ->id; j < targ->height; j += targ->thread_count) {
        double y0 = j * ((targ->upper - targ->lower) / targ->height) + targ->lower;
        v_y0 = _mm512_set1_pd(y0);

        int i;
        for (i = 0; i + 7 < targ->width; i += 8) {
            v_x0 = _mm512_set_pd(
                (i + 7) * x0_step + targ->left,
                (i + 6) * x0_step + targ->left,
                (i + 5) * x0_step + targ->left,
                (i + 4) * x0_step + targ->left,
                (i + 3) * x0_step + targ->left,
                (i + 2) * x0_step + targ->left,
                (i + 1) * x0_step + targ->left,
                i * x0_step + targ->left
            );

            v_x = _mm512_setzero_pd();
            v_y = _mm512_setzero_pd();
            v_repeats = _mm512_setzero_si512();

            mask = 0xFF;  // All 8 bits set
            for (int k = 0; k < targ->iters && mask; ++k) {
                // Compute x and y squares
                v_xx = _mm512_mul_pd(v_x, v_x);
                v_yy = _mm512_mul_pd(v_y, v_y);
                
                // Calculate lengths and update mask
                v_length_squared = _mm512_add_pd(v_xx, v_yy);
                __mmask8 new_mask = _mm512_cmp_pd_mask(v_length_squared, v_four, _CMP_LT_OQ);
                
                // Break if all points have escaped
                if (!new_mask) break;
                
                // Update x and y
                v_xy = _mm512_mul_pd(v_x, v_y);
                v_x = _mm512_add_pd(_mm512_sub_pd(v_xx, v_yy), v_x0);
                v_y = _mm512_fmadd_pd(v_two, v_xy, v_y0);
                
                // Increment repeats where mask is true
                v_repeats = _mm512_mask_add_epi64(v_repeats, new_mask, v_repeats, v_one);
                
                mask = new_mask;
            }

            int64_t repeats[8];
            _mm512_storeu_si512((__m512i*)repeats, v_repeats);

            for (int k = 0; k < 8; ++k) {
                targ->image[j * targ->width + i + k] = (int)repeats[k];
            }
        }

        // Handle remaining pixels sequentially
        for (; i < targ->width; ++i) {
            double x0 = i * x0_step + targ->left;
            mandelbrot_calculate_point(x0, y0, targ->iters, &targ->image[j * targ->width + i]);
        }
    }

    return NULL;
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int num_cpus = CPU_COUNT(&cpu_set);

    /* argument parsing */
    // assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    // assert(image);

    /* prepare thread arguments */
    int num_threads = num_cpus;  // Use number of CPUs as number of threads
    pthread_t threads[num_threads];
    thread_arg_t thread_args[num_threads];

    /* create threads */
    for (int i = 0; i < num_threads; ++i) {
        thread_args[i] = (thread_arg_t){
            .id = i,
            .thread_count = num_threads,
            .width = width,
            .height = height,
            .iters = iters,
            .left = left,
            .right = right,
            .lower = lower,
            .upper = upper,
            .image = image
        };
        pthread_create(&threads[i], NULL, mandelbrot_thread, &thread_args[i]);
    }

    /* wait for threads to finish */
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);

    return 0;
}