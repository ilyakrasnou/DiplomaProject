__kernel void two_matrix_mul(const int M,
                             const int N,
                             const int K,
                             const int P,
                             const __global float* A, // M*K
                             const __global float* B, // K*N
                             const __global float* C, // N*P
                             __global float* X, // M*N
                             __global float* Y ) { // M*P
    const int globalRow = get_group_id(0); // < M
    const int globalCol = get_local_id(0); // < max(N, P)
    float acc = 0.0f;

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

// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
__kernel void two_matrix_mul_os_is(const int M,
                                   const int N,
                                   const int K,
                                   const int P,
                                   const __global float* A, // M*K
                                   const __global float* B, // K*N
                                   const __global float* C, // N*P
                                   __global float* X, // M*N
                                   __global float* Y, // M*P 
                                   __local float* buffer) { // P*N

    const int globalRow = get_group_id(0); // < M
    const int globalCol = get_local_id(0); // < max(N, P)
    float acc = 0.0f;

    // __local int block[0];

    // if (globalCol == 0) {
    //     atomic_xchg(block, 0);
    // }

    // barrier(CLK_LOCAL_MEM_FENCE);

    if (globalCol < N) {
        // calculate X[m, n]
        for (int k = 0; k < K; ++k) {
            acc += A[globalRow * K + k] * B[k * N + globalCol];
        }

        // write X[m, n]
        // X[globalRow * N + globalCol] = acc;
        // barrier(CLK_GLOBAL_MEM_FENCE);

        for (int p = 0; p < P; ++p) {
            // get mutex
            // while (atomic_cmpxchg(block, 0, 1) != 0) { }

            // make multiplication X[m,n] * C[n,p] and write to group buffer 
            buffer[p * N + globalCol] = acc * C[globalCol * P + p];

            // release_mutex(&block[p]);
            // int prevVal = atomic_cmpxchg(block, 1, 0);
        }
        // write X[m, n]
        X[globalRow * N + globalCol] = acc;

    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (globalCol < P) {
        acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            acc += buffer[globalCol * N + n];
        }
        Y[globalRow * P + globalCol] = acc;
    }
}