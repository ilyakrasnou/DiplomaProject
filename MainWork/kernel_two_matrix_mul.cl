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


__kernel void two_matrix_mul_os_is(const int M,
                                   const int N,
                                   const int K,
                                   const int P,
                                   const __global float* A, // M*K
                                   const __global float* B, // K*N
                                   const __global float* C, // N*P
                                   __global float* X, // M*N
                                   __global float* Y, // M*P 
                                   __local float* buffer) { // N

    const int globalRow = get_group_id(0); // < M
    const int globalCol = get_local_id(0); // < N
    float acc = 0.0f;
    float bufferSum = 0.0f;

    // calculate X[m, n]
    for (int k = 0; k < K; ++k) {
        acc += A[globalRow * K + k] * B[k * N + globalCol];
    }

    for (int p = 0; p < P; ++p) {
        // make multiplication X[m,n] * C[n,p] and write to group buffer 
        buffer[globalCol] = acc * C[globalCol * P + p];

        // sync group buffer to calculate reduced sum
        barrier(CLK_LOCAL_MEM_FENCE);

        // calculate Y[m, p]
        // first thread calculate reduced sum over all group buffer
        if (globalCol == 0) {
            bufferSum = 0.0f;
            for (int n = 0; n < N; ++n) {
                bufferSum += buffer[n]; 
            }
            // write sum to global memory
            Y[globalRow * P + p] = bufferSum;
        }
        // sync group buffer to write in it on next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write X[m, n]
    X[globalRow * N + globalCol] = acc;
}