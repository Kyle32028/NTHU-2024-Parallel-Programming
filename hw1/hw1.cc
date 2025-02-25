#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <boost/sort/spreadsort/spreadsort.hpp>

void oddEvenSort(float*& local_data, int local_n, int rank, int size, float*& partner_data, float*& merged_data) {
    MPI_Status status;

    boost::sort::spreadsort::spreadsort(local_data, local_data + local_n);

    for (int phase = 0; phase <= size; ++phase) {
        int partner;
        if (phase % 2 == 0) {
            // Even phase
            partner = (rank % 2 == 0) ? rank + 1 : rank - 1;
        } else {
            // Odd phase
            partner = (rank % 2 == 0) ? rank - 1 : rank + 1;
        }

        // Adjust partner if it's invalid
        if (partner < 0 || partner >= size) {
            partner = MPI_PROC_NULL;
        }

        if (partner != MPI_PROC_NULL) {
            // Exchange boundary elements to check if merge is necessary
            float local_first, local_last;
            float partner_boundary_first, partner_boundary_last;
            int partner_n = 0;

            // Get partner's local_n
            MPI_Sendrecv(&local_n, 1, MPI_INT, partner, phase,
                        &partner_n, 1, MPI_INT, partner, phase,
                        MPI_COMM_WORLD, &status);

            if (rank < partner) { // left rank
                local_last = local_data[local_n - 1];
                MPI_Sendrecv(&local_last, 1, MPI_FLOAT, partner, phase + 1,
                             &partner_boundary_first, 1, MPI_FLOAT, partner, phase + 1,
                             MPI_COMM_WORLD, &status);

                if (local_last <= partner_boundary_first) {
                    // Skip merge
                    continue;
                }
                
                local_first = local_data[0];
                MPI_Sendrecv(&local_first, 1, MPI_FLOAT, partner, phase + 2,
                             &partner_boundary_last, 1, MPI_FLOAT, partner, phase + 2,
                             MPI_COMM_WORLD, &status);
                
                if (local_first >= partner_boundary_last) {
                    // Exchange data
                    MPI_Sendrecv(local_data, local_n, MPI_FLOAT, partner, phase + 3,
                                 partner_data, partner_n, MPI_FLOAT, partner, phase + 3,
                                 MPI_COMM_WORLD, &status);

                    if (local_n == partner_n) {
                        std::memcpy(local_data, partner_data, local_n * sizeof(float));
                        continue;
                    } else { // local_n - partner_n == 1
                        std::memcpy(local_data, partner_data, partner_n * sizeof(float));
                        local_data[local_n - 1] = local_first;
                        continue;
                    }
                }
            } else { // right rank
                local_first = local_data[0];
                MPI_Sendrecv(&local_first, 1, MPI_FLOAT, partner, phase + 1,
                            &partner_boundary_last, 1, MPI_FLOAT, partner, phase + 1,
                            MPI_COMM_WORLD, &status);

                if (local_first >= partner_boundary_last) {
                    // Skip merge
                    continue;
                }

                local_last = local_data[local_n - 1];
                MPI_Sendrecv(&local_last, 1, MPI_FLOAT, partner, phase + 2,
                            &partner_boundary_first, 1, MPI_FLOAT, partner, phase + 2,
                            MPI_COMM_WORLD, &status);
                
                if (local_last <= partner_boundary_first) {
                    // Exchange data
                    MPI_Sendrecv(local_data, local_n, MPI_FLOAT, partner, phase + 3,
                                partner_data, partner_n, MPI_FLOAT, partner, phase + 3,
                                MPI_COMM_WORLD, &status);

                    if (local_n == partner_n) {
                        std::memcpy(local_data, partner_data, local_n * sizeof(float));
                        continue;
                    } else { // local_n - partner_n == -1
                        std::memcpy(local_data, partner_data + 1, local_n * sizeof(float));
                        continue;
                    }
                }
            }

            // Exchange data
            MPI_Sendrecv(local_data, local_n, MPI_FLOAT, partner, phase + 2,
                        partner_data, partner_n, MPI_FLOAT, partner, phase + 2,
                        MPI_COMM_WORLD, &status);

            int i = 0, j = 0;
            if (rank < partner) {
                // Keep the smallest local_n elements
                int count = 0;
                while (count < local_n) {
                    if (local_data[i] < partner_data[j]) {
                        merged_data[count++] = local_data[i++];
                    } else {
                        merged_data[count++] = partner_data[j++];
                    }
                }
            } else {
                // Keep the largest local_n elements
                int count = local_n - 1;
                i = local_n - 1;
                j = partner_n - 1;
                while (count >= 0) {
                    if (local_data[i] > partner_data[j]) {
                        merged_data[count--] = local_data[i--];
                    } else {
                        merged_data[count--] = partner_data[j--];
                    }
                }
            }

            // Update local_data to point to merged_data
            float* temp_ptr = local_data;
            local_data = merged_data;
            merged_data = temp_ptr;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // if (argc != 4) {
    //     if (rank == 0) {
    //         std::cerr << "Usage: " << argv[0] << " <n> <input_file> <output_file>" << std::endl;
    //     }
    //     MPI_Finalize();
    //     return 1;
    // }

    int n = std::atoi(argv[1]);
    const char* input_file = argv[2];
    const char* output_file = argv[3];

    int local_n = n / size;
    int remainder = n % size;
    if (rank < remainder) {
        local_n++;
    }

    float* local_data = (float*)malloc(local_n * sizeof(float));
    float* partner_data = (float*)malloc((local_n + 1) * sizeof(float));
    float* merged_data = (float*)malloc(local_n * sizeof(float));

    // Read input file using MPI-IO
    MPI_File in_file;
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &in_file);

    MPI_Offset offset = rank * (n / size) * sizeof(float);
    if (rank < remainder) {
        offset += rank * sizeof(float);
    } else {
        offset += remainder * sizeof(float);
    }

    MPI_File_read_at(in_file, offset, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&in_file);

    // Perform odd-even sort
    oddEvenSort(local_data, local_n, rank, size, partner_data, merged_data);

    // Write output file using MPI-IO
    MPI_File out_file;
    MPI_File_open(MPI_COMM_WORLD, output_file, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out_file);

    MPI_File_write_at(out_file, offset, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&out_file);

    free(local_data);
    free(partner_data);
    free(merged_data);

    MPI_Finalize();
    return 0;
}