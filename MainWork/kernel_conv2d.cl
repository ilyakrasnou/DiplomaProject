int id(int c, int x, int y, int C, int X, int Y) {
    return x + X * (y + Y * c);
}

int f_id(int a, int c, int x, int y, int A, int C, int X, int Y) {
    return x + X * (y + Y * (c + C * a));
}

float ReLU(float value) {
    // return value > 0 ? value : 0;
    return max(0.0f, value);
}

__kernel void conv2D_tranform(int N1y, int N1x, int C1,
                              int N2y, int N2x, int C2,
                              int Fy, int Fx, 
                              const __global float *I,
                              const __global float *F,
                            //   const __global float *B,
                              __global float *O) {
    const int n2y = get_group_id(0);
    const int n2x = get_local_id(0);

    for (int c2 = 0; c2 < C2; c2++) {
    // for (int n2y = 0; n2y < N2y; n2y++)
    // for (int n2x = 0; n2x < N2x; n2x++) {

        // output width
        // O[id(c2, n2x, n2y, C2, N2x, N2y)] = B[c2];
        // O[id(c2, n2x, n2y, C2, N2x, N2y)] = 0;
        float t = 0;

        for (int c1 = 0; c1 < C1; c1++)
        for (int fy = 0; fy < Fy; fy++)
        for (int fx = 0; fx < Fx; fx++) {
            t += I[id(c1, n2x+fx, n2y+fy, C1, N1x, N1y)] * F[f_id(c2, c1, fx, fy, C2, C1, Fx, Fy)];
        }
        O[id(c2, n2x, n2y, C2, N2x, N2y)] = ReLU(t);
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
    const int n3y = get_group_id(0);
    const int n3x = get_local_id(0);

    for (int c3 = 0; c3 < C3; c3++) {
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
}