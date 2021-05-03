#define BSIZE 16
#define id(c, x, y, C, X, Y) ((x) + (X) * ((y) + (Y) * (c)))
#define f_id(a, c, x, y, A, C, X, Y) ((x) + (X) * ((y) + (Y) * ((c) + (C) * (a))))
#define ReLU(v) ((v) > 0.0f ? (v) : 0.0f)
#define ReLU(v) (max((v), 0.0f))

// int id(int c, int x, int y, int C, int X, int Y) {
//     return x + X * (y + Y * c);
// }

// int f_id(int a, int c, int x, int y, int A, int C, int X, int Y) {
//     return x + X * (y + Y * (c + C * a));
// }

// float ReLU(float value) {
//     // return value > 0 ? value : 0;
//     return max(0.0f, value);
// }

__kernel void conv2D_tranform(int N1y, int N1x, int C1,
                              int N2y, int N2x, int C2,
                              int Fy, int Fx, 
                              __global float *I,
                              __global float *F,
                            //__global float *B,
                              __global float *O) {
    const int ty = get_group_id(0); // < N2y / BSIZE
    const int tx = get_group_id(1); // < N2x / BSIZE
    const int c2 = get_local_id(0);

    const int n2y_bound = min(N2y, (ty+1)*BSIZE);
    const int n2x_bound = min(N2x, (tx+1)*BSIZE);

    const int n2y_0 = ty * BSIZE;
    const int n2x_0 = tx * BSIZE;

    float t;

    for (int n2y = n2y_0; n2y < n2y_bound; ++n2y)
    for (int n2x = n2x_0; n2x < n2x_bound; ++n2x) {
        t = 0;

        for (int c1 = 0; c1 < C1; ++c1) 
        for (int fy = 0; fy < Fy; ++fy) 
        for (int fx = 0; fx < Fx; ++fx) {
            t += I[id(c1, n2x+fx, n2y+fy, C1, N1x, N1y)] * F[f_id(c2, c1, fx, fy, C2, C1, Fx, Fy)];
        }
               
        O[id(c2, n2x, n2y, C2, N2x, N2y)] = ReLU(t);        
    }
}

__kernel void two_conv2D_fusion(int N1y, int N1x, int C1,
                                int N2y, int N2x, int C2,
                                int N3y, int N3x, int C3,
                                int F1y, int F1x, 
                                int F2y, int F2x, 
                                int Tx,  int Ty,
                                const __global float *I,
                                const __global float *F1,
                                //   const __global float *B1,
                                __global float *O1,
                                const __global float *F2,
                                // const __global float *B2,
                                __global float *O2) {    
    const int ty = get_group_id(0); // < N3y / Tx
    const int tx = get_group_id(1); // < N3x / Ty
    const int c3 = get_local_id(0); // < C3

    const int n3y_bound = min(N3y, (ty+1)*Ty);
    const int n3x_bound = min(N3x, (tx+1)*Ty);

    const int n3y_0 = ty * Ty;
    const int n3x_0 = tx * Tx;
    float t = 0;
    float buffer[BSIZE][BSIZE];

    // for (int by = 0; by < n2y_bound - n2y_0 - F2y + 1; ++by)
    // for (int bx = 0; bx < n2x_bound - n2x_0 - F2x + 1; ++bx) {
    //     O2[id(c3, n2x_0 + bx, n2y_0 + by, C3, N3x, N3y)] = 0;   
    // }

    for (int c2 = 0; c2 < C2; ++c2) {
        // calculate intermediate layer
        for (int n2y = n3y_0; n2y < n3y_bound + F2y - 1; ++n2y)
        for (int n2x = n3x_0; n2x < n3x_bound + F2x - 1; ++n2x) {
            t = 0;
        
            for (int c1 = 0; c1 < C1; ++c1)
            for (int f1y = 0; f1y < F1y; ++f1y)
            for (int f1x = 0; f1x < F1x; ++f1x) {
                t += I[id(c1, n2x+f1x, n2y+f1y, C1, N1x, N1y)] * F1[f_id(c2, c1, f1x, f1y, C2, C1, F1x, F1y)];
            }
        
            buffer[n2y-n3y_0][n2x-n3x_0] = ReLU(t);
        }

        for (int by = 0; by < n3y_bound - n3y_0; ++by)
        for (int bx = 0; bx < n3x_bound - n3x_0; ++bx) {
            t = 0;

            for (int f2y = 0; f2y < F2y; ++f2y)
            for (int f2x = 0; f2x < F2x; ++f2x) {
                t += buffer[by+f2y][bx+f2x] * F2[f_id(c3, c2, f2x, f2y, C3, C2, F2x, F2y)];
            }

            if (c2 == 0)
                O2[id(c3, n3x_0 + bx, n3y_0 + by, C3, N3x, N3y)] = 0;

            O2[id(c3, n3x_0 + bx, n3y_0 + by, C3, N3x, N3y)] += t;   
        }
    }

    for (int by = 0; by < n3y_bound - n3y_0; ++by)
    for (int bx = 0; bx < n3x_bound - n3x_0; ++bx) {
        O2[id(c3, n3x_0 + bx, n3y_0 + by, C3, N3x, N3y)] = ReLU(O2[id(c3, n3x_0 + bx, n3y_0 + by, C3, N3x, N3y)]);   
    }
}

__kernel void two_conv2D_tranform(int N1y, int N1x, int C1,
                                  int N2y, int N2x, int C2,
                                  int N3y, int N3x, int C3,
                                  int F1y, int F1x, 
                                  int F2y, int F2x, 
                                  const __global float *I,
                                  const __global float *F1,
                                  //   const __global float *B1,
                                  __global float *O1,
                                  const __global float *F2,
                                  // const __global float *B2,
                                  __global float *O2) {
    const int n3y = get_group_id(0); // < N3y
    const int n3x = get_group_id(1); // < N3x
    const int c3 = get_local_id(0);

    float o = 0;

    for (int c2 = 0; c2 < C2; c2++)
    for (int n2y = n3y; n2y < n3y + F2y; n2y++)
    for (int n2x = n3x; n2x < n3x + F2x; n2x++) {
        // output width
        // O[id(c2, n2x, n2y, C2, N2x, N2y)] = B[c2];
        // O[id(c2, n2x, n2y, C2, N2x, N2y)] = 0;
        float t = 0;

        for (int c1 = 0; c1 < C1; c1++)
        for (int f1y = 0; f1y < F1y; f1y++)
        for (int f1x = 0; f1x < F1x; f1x++) {
            t += I[id(c1, n2x+f1x, n2y+f1y, C1, N1x, N1y)] * F1[f_id(c2, c1, f1x, f1y, C2, C1, F1x, F1y)];
        }
        
        t = ReLU(t);
        O1[id(c2, n2x, n2y, C2, N2x, N2y)] = t;

        int f2y = n2y - n3y, f2x = n2x - n3x;
        o += t * F2[f_id(c3, c2, f2x, f2y, C3, C2, F2x, F2y)];
        // for (int n3y = max(0, n2y-F2y+1); n3y < N3 && n3y <= n2y; ++n3y)
        // for (int n3x = max(0, n2x-F2x+1); n3x < N3 && n3x <= n2x; ++n3x) 
        // for (int c3 = 0; c3 < C3; ++c3) {
            // int f2y = n2y - n3y, f2x = n2x - n3x;
            // need atomic
        //     O2[id(c3, n3x, n3y, C3, N2x, N2y)] += t * F2[f_id(c3, c2, f2x, f2y, C3, C2, F2x, F2y)];
        // }
    }

    O2[id(c3, n3x, n3y, C3, N3x, N3y)] = ReLU(o);
}

__kernel void two_conv2D_tranform_buffer(int N1y, int N1x, int C1,
                                         int N2y, int N2x, int C2,
                                         int N3y, int N3x, int C3,
                                         int F1y, int F1x, 
                                         int F2y, int F2x,
                                         const __global float *I,
                                         const __global float *F1,
                                         //   const __global float *B1,
                                         __global float *O1,
                                         const __global float *F2,
                                         // const __global float *B2,
                                         __global float *O2) {
    const int ty = get_group_id(0); // < N3y / 32 
    const int tx = get_group_id(1); // < N3x / 32
    const int c3 = get_local_id(0);

    const int n3y_bound = min(N3y, (ty+1)*BSIZE);
    const int n3x_bound = min(N3x, (tx+1)*BSIZE);

    const int n3y_0 = ty * BSIZE;
    const int n3x_0 = tx * BSIZE;

    if (n3y_0 >= N3y || n3x_0 >= N3x)
        return;

    float buffer[BSIZE][BSIZE];
    float t;

    // clear buffer
    for (int i = 0; i < n3y_bound - n3y_0; ++i)
    for (int j = 0; j < n3x_bound - n3x_0; ++j)
        buffer[i][j] = 0;

    // calculate buffer
    for (int c2 = 0; c2 < C2; c2++)
    for (int n2y = n3y_0; n2y < n3y_bound + F2y; n2y++) 
    for (int n2x = n3x_0; n2x < n3x_bound + F2x; n2x++) {
        // O[id(c2, n2x, n2y, C2, N2x, N2y)] = B[c2];
        // O[id(c2, n2x, n2y, C2, N2x, N2y)] = 0;
        t = 0;
        
        // calculate value for intermediate layer
        for (int c1 = 0; c1 < C1; c1++)
        for (int f1y = 0; f1y < F1y; f1y++)
        for (int f1x = 0; f1x < F1x; f1x++) {
            t += I[id(c1, n2x+f1x, n2y+f1y, C1, N1x, N1y)] * F1[f_id(c2, c1, f1x, f1y, C2, C1, F1x, F1y)];
        }
        t = ReLU(t);
        // O1[id(c2, n2x, n2y, C2, N2x, N2y)] = t;

        // update values in buffer
        // for (int c3 = 0; c3 < C3; ++c3)
        for (int n3y = max(n3y_0, n2y-F2y+1); n3y < n3y_bound && n3y <= n2y; ++n3y)
        for (int n3x = max(n3x_0, n2x-F2x+1); n3x < n3x_bound && n3x <= n2x; ++n3x) {
            int f2y = n2y - n3y, f2x = n2x - n3x;
            buffer[n3y - n3y_0][n3x - n3x_0] += t * F2[f_id(c3, c2, f2x, f2y, C3, C2, F2x, F2y)];
        }
    }

    // write buffer
    for (int i = 0; i < n3y_bound - n3y_0; ++i)
    for (int j = 0; j < n3x_bound - n3x_0; ++j)
        O2[id(c3, n3x_0 + j, n3y_0 + i, C3, N3x, N3y)] = ReLU(buffer[i][j]);    
}