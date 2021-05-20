#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <vector>

#define id(c, x, y, C, X, Y) ((x) + (X) * ((y) + (Y) * (c)))
#define f_id(a, c, x, y, A, C, X, Y) ((x) + (X) * ((y) + (Y) * ((c) + (C) * (a))))
#define ReLU(v) (max((v), 0.0f))

const int T = 1;

__global__ void convolution(int N1y, int N1x, int C1,
                            int N2y, int N2x, int C2,
                            int Fy, int Fx,
                            int Ty, int Tx,
                            const float *I,
                            const float *F,
                            // const float *B,
                            float *O) {
    const int ty = blockIdx.x; // < N2y / Ty
    const int tx = blockIdx.y; // < N2x / Tx
    const int c2 = threadIdx.x;

    const int n2y_bound = min(N2y, (ty + 1)*Ty);
    const int n2x_bound = min(N2x, (tx + 1)*Tx);

    const int n2y_0 = ty * Ty;
    const int n2x_0 = tx * Tx;

    float t;

    //for (int c2 = 0; c2 < C2; ++c2)
    for (int n2y = n2y_0; n2y < n2y_bound; ++n2y)
    for (int n2x = n2x_0; n2x < n2x_bound; ++n2x) {
        t = 0;

        for (int c1 = 0; c1 < C1; ++c1)
        for (int fx = 0; fx < Fx; ++fx) {
        for (int fy = 0; fy < Fy; ++fy)
            t += I[id(c1, n2x + fx, n2y + fy, C1, N1x, N1y)] * F[f_id(c2, c1, fx, fy, C2, C1, Fx, Fy)];
        }

        O[id(c2, n2x, n2y, C2, N2x, N2y)] = ReLU(t);
    }
}

__global__ void convolution_separable(int N1y, int N1x, int C1,
                                      int N2y, int N2x, int C2,
                                      int Fy, int Fx,
                                      int Ty, int Tx,
                                      const float *I,
                                      const float *F,
                                      // const float *B,
                                      float *O) {
    const int ty = blockIdx.x; // < N2y / Ty
    const int tx = blockIdx.y; // < N2x / Tx
    const int c2 = threadIdx.x;

    const int n2y_bound = min(N2y, (ty + 1)*Ty);
    const int n2x_bound = min(N2x, (tx + 1)*Tx);

    const int n2y_0 = ty * Ty;
    const int n2x_0 = tx * Tx;

    for (int n2y = n2y_0; n2y < n2y_bound; ++n2y)
    for (int n2x = n2x_0; n2x < n2x_bound; ++n2x) {
        float t = 0;

        for (int fy = 0; fy < Fy; ++fy)
        for (int fx = 0; fx < Fx; ++fx) {
            t += I[id(c2, n2x + fx, n2y + fy, C1, N1x, N1y)] * F[f_id(c2, c2, fx, fy, C2, C1, Fx, Fy)];
        }

        O[id(c2, n2x, n2y, C2, N2x, N2y)] = ReLU(t);
    }
}

__global__ void convolution_one_to_one(int N1y, int N1x, int C1,
                                       int N2y, int N2x, int C2,
                                       int Fy, int Fx,
                                       int Ty, int Tx,
                                       const float *I,
                                       const float *F,
                                       // const float *B,
                                       float *O) {
    const int ty = blockIdx.x; // < N2y / Ty
    const int tx = blockIdx.y; // < N2x / Tx
    const int c2 = threadIdx.x;

    const int n2y_bound = min(N2y, (ty + 1)*Ty);
    const int n2x_bound = min(N2x, (tx + 1)*Tx);

    const int n2y_0 = ty * Ty;
    const int n2x_0 = tx * Tx;

    float t;

    for (int n2y = n2y_0; n2y < n2y_bound; ++n2y)
    for (int n2x = n2x_0; n2x < n2x_bound; ++n2x) {
        t = 0;

        for (int c1 = 0; c1 < C1; ++c1) {
            t += I[id(c1, n2x, n2y, C1, N1x, N1y)] * F[f_id(c2, c1, 0, 0, C2, C1, Fx, Fy)];
        }

        O[id(c2, n2x, n2y, C2, N2x, N2y)] = ReLU(t);
    }
}

extern __shared__ float buffer[];

__global__ void convolution_dep_sep_fused(int N1y, int N1x, int C1,
                                          int N2y, int N2x, int C2,
                                          int N3y, int N3x, int C3,
                                          int F1y, int F1x,
                                          int F2y, int F2x,
                                          int Ty, int Tx,
                                          const float *I,
                                          const float *F1,
                                          //   const float *B1,
                                          float *O1,
                                          const float *F2,
                                          // const float *B2,
                                          float *O2) {
    const int ty_0 = blockIdx.x * Ty;
    const int tx_0 = blockIdx.y * Tx;
    const int c0 = threadIdx.x;

    const int ty_bound = min(N3y - ty_0, Ty);
    const int tx_bound = min(N3x - tx_0, Tx);

    float t;

    if (c0 < C2) {
        for (int by = 0; by < ty_bound + F2y - 1; ++by)
        for (int bx = 0; bx < tx_bound + F2x - 1; ++bx) {
            float t = 0;

            // calculate value for intermediate layer
            for (int f1y = 0; f1y < F1y; f1y++)
            for (int f1x = 0; f1x < F1x; f1x++) {
                t += I[id(c0, tx_0 + bx + f1x, ty_0 + by + f1y, C1, N1x, N1y)] * F1[f_id(c0, c0, f1x, f1y, C2, C1, F1x, F1y)];
            }

            buffer[id(c0, bx, by, C2, Tx + F2x - 1, Ty + F2y - 1)] = ReLU(t);
        }
    }

    __syncthreads();

    if (c0 < C3) {
        for (int by = 0; by < ty_bound; by += 1)
        for (int bx = 0; bx < tx_bound; bx += 1) {
            float t = 0;

            for (int c2 = 0; c2 < C2; c2++) {
                t += buffer[id(c2, bx, by, C2, Tx + F2x - 1, Ty + F2y - 1)] * F2[f_id(c0, c2, 0, 0, C3, C2, F2x, F2y)];
            }

            O2[id(c0, tx_0 + bx, ty_0 + by, C3, N3x, N3y)] = ReLU(t);
        }
    }
}

bool float_compare(float lhs,
                   float rhs,
                   float eps) {
    return fabs(lhs - rhs) <= eps;
}

bool compare_results(const std::vector<float>& A,
                     const std::vector<float>& B,
                     float eps = 1e-7) {
    if (A.size() != B.size())
        return false;

    for (int i = 0; i < A.size(); i++) {
        if (!float_compare(A[i], B[i], eps)) {
            std::cout << i << " " << A[i] << " " << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool compare_convolution(int Ox, int Oy,
                         int Fx, int Fy,
                         int C1, int C2,
                         const std::vector<float>& I,
                         const std::vector<float>& F,
                         const std::vector<float>& O,
                         float eps = 1e-3) {
    for (int c2 = 0; c2 < C2; ++c2)
    for (int oy = 0; oy < Oy; ++oy)
    for (int ox = 0; ox < Ox; ++ox) {
        float val = 0.0f;

        for (int c1 = 0; c1 < C1; ++c1)
        for (int fy = 0; fy < Fy; ++fy)
        for (int fx = 0; fx < Fx; ++fx) {
            val += I[id(c1, ox + fx, oy + fy, C1, Ox + Fx - 1, Oy + Fy - 1)] * F[f_id(c2, c1, fx, fy, C2, C1, Fx, Fy)];
        }

        val = ReLU(val);

        // std::cout << c2 << " " << ox << " " << oy << " " << O[id(c2, ox, oy, C2, Ox, Oy)] << " " << val << std::endl; 
        if (!float_compare(O[id(c2, ox, oy, C2, Ox, Oy)], val, eps)) {
            std::cout << c2 << " " << ox << " " << oy << " " << O[id(c2, ox, oy, C2, Ox, Oy)] << " " << val << std::endl;
            return false;
        }
    }

    return true;
}

bool check_error_status(cudaError_t status, const char *error_message) {
    if (status != cudaSuccess) {
        fprintf(stderr, error_message);
        return true;
    }

    return false;
}

cudaError_t make_two_convolution(float *A,
                                 float *Filter1,
                                 float *C,
                                 float *Filter2,
                                 float *E,
                                 int n1, int c1,
                                 int n2, int c2,
                                 int n3, int c3,
                                 int f1, int f2)
{
    float *dev_A = 0;
    float *dev_F1 = 0;
    float *dev_C = 0;
    float *dev_F2 = 0;
    float *dev_E = 0;
    cudaError_t cudaStatus;

    // Time measurement
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    cudaEventRecord(startGPU, 0);


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (check_error_status(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n"))
        goto Error;


    // Allocate buffers on GPU
    cudaStatus = cudaMalloc((void**)&dev_A, c1*n1*n1 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;

    cudaStatus = cudaMalloc((void**)&dev_F1, c2*c1*f1*f1 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;

    cudaStatus = cudaMalloc((void**)&dev_C, c2*n2*n2 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;

    cudaStatus = cudaMalloc((void**)&dev_F2, c3*c2*f2*f2 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;

    cudaStatus = cudaMalloc((void**)&dev_E, c3*n3*n3 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;


    // Copy input data from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_A, A, c1*n1*n1 * sizeof(float), cudaMemcpyHostToDevice);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    cudaStatus = cudaMemcpy(dev_F1, Filter1, c2*c1*f1*f1 * sizeof(float), cudaMemcpyHostToDevice);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    cudaStatus = cudaMemcpy(dev_F2, Filter2, c3*c2*f2*f2 * sizeof(float), cudaMemcpyHostToDevice);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    // Run first convolution processing

    dim3 dimBlock1(c2, 1);
    dim3 dimGrid1((n2 + T - 1) / T, (n2 + T - 1) / T);
    // Launch a kernel on the GPU with one thread for each element.
    convolution<<<dimGrid1, dimBlock1>>>(n1, n1, c1,
                                         n2, n2, c2,
                                         f1, f1,
                                         T, T,
                                         dev_A, dev_F1, dev_C);

    //// Check for any errors launching the kernel
    //cudaStatus = cudaGetLastError();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //    goto Error;
    //}

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Run second convolution processing

    dim3 dimBlock2(c3, 1);
    dim3 dimGrid2((n3 + T - 1) / T, (n3 + T - 1) / T);
    // Launch a kernel on the GPU with one thread for each element.
    convolution<<<dimGrid2, dimBlock2>>>(n2, n2, c2,
                                         n3, n3, c3,
                                         f2, f2,
                                         T, T,
                                         dev_C, dev_F2, dev_E);

    //// Check for any errors launching the kernel
    //cudaStatus = cudaGetLastError();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //    goto Error;
    //}

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }


    // Copy output result from GPU
    cudaStatus = cudaMemcpy(C, dev_C, c2*n2*n2 * sizeof(float), cudaMemcpyDeviceToHost);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    cudaStatus = cudaMemcpy(E, dev_E, c3*n3*n3 * sizeof(float), cudaMemcpyDeviceToHost);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    cudaEventRecord(stopGPU, 0);
    cudaEventSynchronize(stopGPU);
    float elapsedTimeGPU;
    cudaEventElapsedTime(&elapsedTimeGPU, startGPU, stopGPU);
    fprintf(stdout, "Elapsed GPU time: %.3f\n", elapsedTimeGPU);
    fflush(stdout);

Error:
    cudaFree(dev_E);
    cudaFree(dev_F2);
    cudaFree(dev_C);
    cudaFree(dev_F1);
    cudaFree(dev_A);

    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);

    return cudaStatus;
}


cudaError_t make_two_layer_dep_sep(float *A,
                                   float *Filter1,
                                   float *C,
                                   float *Filter2,
                                   float *E,
                                   int n1, int c1,
                                   int n2, int c2,
                                   int n3, int c3,
                                   int f1, int f2)
{
    float *dev_A = 0;
    float *dev_F1 = 0;
    float *dev_C = 0;
    float *dev_F2 = 0;
    float *dev_E = 0;
    cudaError_t cudaStatus;

    // Time measurement
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    cudaEventRecord(startGPU, 0);


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (check_error_status(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n"))
        goto Error;


    // Allocate buffers on GPU
    cudaStatus = cudaMalloc((void**)&dev_A, c1*n1*n1 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;

    cudaStatus = cudaMalloc((void**)&dev_F1, c2*c1*f1*f1 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;

    cudaStatus = cudaMalloc((void**)&dev_C, c2*n2*n2 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;

    cudaStatus = cudaMalloc((void**)&dev_F2, c3*c2*f2*f2 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;

    cudaStatus = cudaMalloc((void**)&dev_E, c3*n3*n3 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;


    // Copy input data from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_A, A, c1*n1*n1 * sizeof(float), cudaMemcpyHostToDevice);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    cudaStatus = cudaMemcpy(dev_F1, Filter1, c2*c1*f1*f1 * sizeof(float), cudaMemcpyHostToDevice);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    cudaStatus = cudaMemcpy(dev_F2, Filter2, c3*c2*f2*f2 * sizeof(float), cudaMemcpyHostToDevice);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    // Run first convolution processing

    dim3 dimBlock1(c2, 1);
    dim3 dimGrid1((n2 + T - 1) / T, (n2 + T - 1) / T);
    // Launch a kernel on the GPU with one thread for each element.
    convolution_separable<<<dimGrid1, dimBlock1>>>(n1, n1, c1,
                                                   n2, n2, c2,
                                                   f1, f1,
                                                   T, T,
                                                   dev_A, dev_F1, dev_C);

    //// Check for any errors launching the kernel
    //cudaStatus = cudaGetLastError();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //    goto Error;
    //}

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Run second convolution processing

    dim3 dimBlock2(c3, 1);
    dim3 dimGrid2((n3 + T - 1) / T, (n3 + T - 1) / T);
    // Launch a kernel on the GPU with one thread for each element.
    convolution_one_to_one<<<dimGrid2, dimBlock2>>>(n2, n2, c2,
                                                    n3, n3, c3,
                                                    f2, f2,
                                                    T, T,
                                                    dev_C, dev_F2, dev_E);

    //// Check for any errors launching the kernel
    //cudaStatus = cudaGetLastError();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //    goto Error;
    //}

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }


    // Copy output result from GPU
    cudaStatus = cudaMemcpy(C, dev_C, c2*n2*n2 * sizeof(float), cudaMemcpyDeviceToHost);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    cudaStatus = cudaMemcpy(E, dev_E, c3*n3*n3 * sizeof(float), cudaMemcpyDeviceToHost);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    cudaEventRecord(stopGPU, 0);
    cudaEventSynchronize(stopGPU);
    float elapsedTimeGPU;
    cudaEventElapsedTime(&elapsedTimeGPU, startGPU, stopGPU);
    fprintf(stdout, "Elapsed GPU time: %.3f\n", elapsedTimeGPU);
    fflush(stdout);

Error:
    cudaFree(dev_E);
    cudaFree(dev_F2);
    cudaFree(dev_C);
    cudaFree(dev_F1);
    cudaFree(dev_A);

    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);

    return cudaStatus;
}

cudaError_t make_dep_sep_fused(float *A,
                               float *Filter1,
                               float *C,
                               float *Filter2,
                               float *E,
                               int n1, int c1,
                               int n2, int c2,
                               int n3, int c3,
                               int f1, int f2)
{
    float *dev_A = 0;
    float *dev_F1 = 0;
    float *dev_C = 0;
    float *dev_F2 = 0;
    float *dev_E = 0;
    cudaError_t cudaStatus;

    // Time measurement
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    cudaEventRecord(startGPU, 0);


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (check_error_status(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n"))
        goto Error;


    // Allocate buffers on GPU
    cudaStatus = cudaMalloc((void**)&dev_A, c1*n1*n1 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;

    cudaStatus = cudaMalloc((void**)&dev_F1, c2*c1*f1*f1 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;

    cudaStatus = cudaMalloc((void**)&dev_C, c2*n2*n2 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;

    cudaStatus = cudaMalloc((void**)&dev_F2, c3*c2*f2*f2 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;

    cudaStatus = cudaMalloc((void**)&dev_E, c3*n3*n3 * sizeof(float));
    if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
        goto Error;


    // Copy input data from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_A, A, c1*n1*n1 * sizeof(float), cudaMemcpyHostToDevice);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    cudaStatus = cudaMemcpy(dev_F1, Filter1, c2*c1*f1*f1 * sizeof(float), cudaMemcpyHostToDevice);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    cudaStatus = cudaMemcpy(dev_F2, Filter2, c3*c2*f2*f2 * sizeof(float), cudaMemcpyHostToDevice);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    // Run first convolution processing

    dim3 dimBlock1(std::max(c2, c3), 1);
    dim3 dimGrid1((n3 + T - 1) / T, (n3 + T - 1) / T);
    //// Launch a kernel on the GPU with one thread for each element.
    convolution_dep_sep_fused<<<dimGrid1, dimBlock1, c2 * (T + f2 - 1) * (T + f2 - 1) * sizeof(float)>>>(n1, n1, c1,
                                                                                                         n2, n2, c2,
                                                                                                         n3, n3, c3,
                                                                                                         f1, f1,
                                                                                                         f2, f2,
                                                                                                         T, T,
                                                                                                         dev_A, dev_F1, dev_C, dev_F2, dev_E);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }


    // Copy output result from GPU
    cudaStatus = cudaMemcpy(C, dev_C, c2*n2*n2 * sizeof(float), cudaMemcpyDeviceToHost);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    cudaStatus = cudaMemcpy(E, dev_E, c3*n3*n3 * sizeof(float), cudaMemcpyDeviceToHost);
    if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
        goto Error;

    cudaEventRecord(stopGPU, 0);
    cudaEventSynchronize(stopGPU);
    float elapsedTimeGPU;
    cudaEventElapsedTime(&elapsedTimeGPU, startGPU, stopGPU);
    fprintf(stdout, "Elapsed GPU time: %.3f\n", elapsedTimeGPU);
    fflush(stdout);

Error:
    cudaFree(dev_E);
    cudaFree(dev_F2);
    cudaFree(dev_C);
    cudaFree(dev_F1);
    cudaFree(dev_A);

    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);

    return cudaStatus;
}


bool test_convolutions() {
    int C1 = 64, C3 = 64, F1 = 3, F2 = 1;
    int C2 = C1;
    // int C1 = 1, C2 = 1, C3 = 1, F1 = 3, F2 = 3;
    // int N1 = rand() % 100 + F1 + F2 + 400;
    int N1 = 100 + F1 + F2;
    // int N1 = rand() % 350 + F1 + F2;
    // int N1 = rand() % 200 + 3, C1 = 1, C2 = 64, C3 = 64, F1 = 3, F2 = 3;
    std::cout << "Start" << std::endl;

    int N2 = N1 - F1 + 1;
    int N3 = N2 - F2 + 1;

    std::cout << N1 << " " << N2 << " " << N3 << std::endl;

    std::vector<float> A(C1*N1*N1);
    std::vector<float> B(C2*C2*F1*F1, 0.0f);
    std::vector<float> C_1(C2*N2*N2);
    std::vector<float> C_2(C2*N2*N2);
    std::vector<float> D(C3*C2*F2*F2);
    std::vector<float> E_1(C3*N3*N3);
    std::vector<float> E_2(C3*N3*N3);

    for (int c = 0; c < C1; c++)
    for (int x = 0; x < N1; x++)
    for (int y = 0; y < N1; y++) {
        A[id(c, x, y, C1, N1, N1)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
    }

    for (int c2 = 0; c2 < C2; c2++)
    for (int x = 0; x < F1; x++)
    for (int y = 0; y < F1; y++) {
        B[f_id(c2, c2, x, y, C2, C2, F1, F1)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
    }

    for (int c3 = 0; c3 < C3; c3++)
    for (int c2 = 0; c2 < C2; c2++)
    for (int x = 0; x < F2; x++)
    for (int y = 0; y < F2; y++) {
        D[f_id(c3, c2, x, y, C3, C2, F2, F2)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
    }

    cudaError cudaStatus;

    std::cout << "Two convolutions " << std::endl;

    cudaStatus = make_two_convolution(A.data(), B.data(), C_1.data(),
                                      D.data(), E_1.data(),
                                      N1, C1, N2, C2, N3, C3, F1, F2);

    if (check_error_status(cudaStatus, "Two convolutions failed!\n"))
        return false;

    std::cout << "Two layer dep sep" << std::endl;

    cudaStatus = make_two_layer_dep_sep(A.data(), B.data(), C_1.data(),
                                        D.data(), E_1.data(),
                                        N1, C1, N2, C2, N3, C3, F1, F2);

    if (check_error_status(cudaStatus, "Two layer dep sep failed!\n"))
        return false;

    std::cout << "Fused layer dep sep" << std::endl;

    cudaStatus = make_dep_sep_fused(A.data(), B.data(), C_2.data(),
                                    D.data(), E_2.data(),
                                    N1, C1, N2, C2, N3, C3, F1, F2);

    if (check_error_status(cudaStatus, "Fused layer dep sep failed!\n"))
        return false;

    bool is_Passed = true;

    is_Passed &= compare_convolution(N2, N2, F1, F1, C1, C2, A, B, C_1, 1e-1);
    is_Passed &= compare_convolution(N3, N3, F2, F2, C2, C3, C_1, D, E_1, 1e-1);

    is_Passed &= compare_results(E_1, E_2, 1e-1);

    return is_Passed;
}


int main()
{
    bool is_Passed = test_convolutions();

    std::cout << "Total: ";
    if (is_Passed) {
        std::cout << "Passed!" << std::endl;
    }
    else {
        std::cout << "Failed!" << std::endl;
    }

    return 0;
}
