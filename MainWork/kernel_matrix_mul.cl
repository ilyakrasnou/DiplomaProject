/*__kernel void matrix_mul(const int M,
                         const int N,
                         const int K,
                         const int TS_x,
                         const int TS_y,
                         const __global float* A,
                         const __global float* B,
                         __global float* C,
                         __local float* A_tile,
                         __local float* B_tile) {

    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    if(globalRow >= M || globalCol >= N) {
        return;
    }

    float acc = 0.0f;
    for (int k=0; k<K; ++k) {
        acc += A[globalRow * K + k] * B[N * k + globalCol];
    }

    C[globalRow*N + globalCol] = acc;
}*

/*#define TS 32
__kernel void matrix_mul(const int M,
                         const int N,
                         const int K,
                         const int TS_x,
                         const int TS_y,
                         const __global float* A,
                         const __global float* B,
                         __global float* C) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = K / TS;
    for (int t = 0; t < numTiles; ++t) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS * t + row;
        const int tiledCol = TS * t + col;
        Asub[row][col] = A[globalRow*K + tiledCol];
        Bsub[row][col] = B[tiledRow*N + globalCol];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k < TS; ++k) {
            acc += Asub[row][k] * Bsub[k][col];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    C[globalRow*N + globalCol] = acc;
}*/

/*__kernel void matrix_mul(const int M,
                         const int N,
                         const int K,
                         const int TS,
                         const int TS_y,
                         const __global float* A,
                         const __global float* B,
                         __global float* C) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = K / TS;
    for (int t=0; t < numTiles; t++) {

        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k < TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    C[globalCol*M + globalRow] = acc;
}*/

__kernel void matrix_mul(const int M,
                         const int N,
                         const int K,
                         const int K1,
                         const int TS,
                         const __global float* A,
                         const __global float* B,
                         __global float* C,
                         __local float* A_tile,
                         __local float* B_tile) {

    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    const int blockRow = get_group_id(0);
    const int blockCol = get_group_id(1);

    int row = blockRow * TS + localRow;
    int col = blockCol * TS + localCol;

    float acc = 0.0f;

    for(int i = 0; i < K / TS; ++i) {

        if(row < M && (i * TS + localCol) < K1) {
            A_tile[localRow * TS + localCol] = A[row * K1 + (i * TS + localCol)];
        }
        else {
            A_tile[localRow * TS + localCol] = 0.0;
        }


        if((i * TS + localRow) < K1 && col < N) {
            B_tile[localRow * TS + localCol] = B[(i * TS + localRow) * N + col];
        }
        else {
            B_tile[localRow * TS + localCol] = 0.0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int j = 0; j < TS; ++j) {
            acc += A_tile[localRow * TS + j] * B_tile[j * TS + localCol];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

    }

    if(row < M && col < N) {
        C[row * N + col] = acc;
    }
}
