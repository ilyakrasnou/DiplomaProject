__kernel void two_matrix_mul(const int M,
                             const int N,
                             const int K,
                             const int P,
                             const __global int* A, // M*K
                             const __global int* B, // K*N
                             const __global int* C, // N*P
                             __global int* X, // M*N
                             __global int* Y ) { // M*P
    const int globalRow = get_group_id(0); // < M
    const int globalCol = get_local_id(0); // < max(N, P)
    int acc = 0.0f;

    if (globalCol < N) {
        acc = 0.0f;
        // calculate X[m, n]
        for (int k = 0; k < K; ++k) {
            acc += A[globalRow * K + k] * B[k * N + globalCol];
        }

        X[globalRow * N + globalCol] = acc;
    }
    // synchronise before next multiplication
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (globalCol < P) {
        acc = 0.0f;
        // calculate Y[m, p]
        for (int n = 0; n < N; ++n) {
            acc += X[globalRow * N + n] * C[n * P + globalCol];
        }

        Y[globalRow * P + globalCol] = acc;
    }
}

__kernel void two_matrix_mul_os_is(const int M,
                                   const int N,
                                   const int K,
                                   const int P,
                                   const __global int* A, // M*K
                                   const __global int* B, // K*N
                                   const __global int* C, // N*P
                                   __global int* X, // M*N
                                   __global int* Y // M*P
                                   ) { 

    const int globalRow = get_group_id(0); // < M
    const int globalCol = get_local_id(0); // < N
    int acc = 0.0f;

    // calculate X[m, n]
    for (int k = 0; k < K; ++k) {
        acc += A[globalRow * K + k] * B[k * N + globalCol];
    }
    for (int p = 0; p < P; ++p) {
        atomic_add(&Y[globalRow * P + p], acc * C[globalCol * P + p]);
    }
    // write X[m, n]
    X[globalRow * N + globalCol] = acc;
}

// __kernel void two_matrix_mul_os_is(const int M,
//                                    const int N,
//                                    const int K,
//                                    const int P,
//                                    const __global int* A, // M*K
//                                    const __global int* B, // K*N
//                                    const __global int* C, // N*P
//                                    __global int* X, // M*N
//                                    __global int* Y // M*P
//                                    ) { 

//     const int globalRow = get_group_id(0); // < M
//     const int globalCol = get_local_id(0); // < max(N, P)
//     int acc = 0.0f;

//     // __local int block[0];

//     // if (globalCol < P) {
//     //     block[globalCol] = 0;
//     //     buffer[globalCol] = 0;
//     // }

//     // barrier(CLK_LOCAL_MEM_FENCE);

//     if (globalCol < N) {
//         // calculate X[m, n]
//         for (int k = 0; k < K; ++k) {
//             acc += A[globalRow * K + k] * B[k * N + globalCol];
//         }

//         // write X[m, n]
//         // X[globalRow * N + globalCol] = acc;
//         // barrier(CLK_GLOBAL_MEM_FENCE);

//         for (int p = 0; p < P; ++p) {
//             // get mutex
//             // while (atomic_cmpxchg(&block[p], 0, 1) != 0) { }

//             // make multiplication X[m,n] * C[n,p] and write to group buffer 
//             // buffer[p] += acc * C[globalCol * P + p];
//             atomic_add(&Y[globalRow * P + p], acc * C[globalCol * P + p]);

//             // release_mutex(&block[p]);
//             // int prevVal = atomic_xchg(&block[p], 0);
//         }
//         // write X[m, n]
//         X[globalRow * N + globalCol] = acc;

//     }

//     // barrier(CLK_LOCAL_MEM_FENCE);
    
//     // if (globalCol < P) {
//         // acc = 0.0f;
//         // for (int n = 0; n < N; ++n) {
//         //     acc += buffer[globalCol * N + n];
//         // }
//         // Y[globalRow * P + globalCol] = buffer[globalCol];
//     // }
// }