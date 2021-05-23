
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>

#include <vector>

#define BSIZE 8

// channel-last
//#define id(c, x, y, C, X, Y) ((c) + (C) * ((x) + (X) * (y)))
//#define f_id(a, c, x, y, A, C, X, Y) ((c) + (C) * ((a) + (A) * ((x) + (X) * (y))))

// channel-first
#define id(c, x, y, C, X, Y) ((x) + (X) * ((y) + (Y) * (c)))
#define f_id(a, c, x, y, A, C, X, Y) ((x) + (X) * ((y) + (Y) * ((c) + (C) * (a))))

#define ReLU(v) (max((v), 0.0f))

__global__ void convolution_simple(int N1y, int N1x, int C1,
	                               int N2y, int N2x, int C2,
	                               int Fy, int Fx,
	                               const float *I,
	                               const float *F,
	                               // const float *B,
	                               float *O) {
	const int ty = blockIdx.x; // < N2y / BSIZE
	const int tx = blockIdx.y; // < N2x / BSIZE
	const int c2 = threadIdx.x;

	const int n2y_bound = min(N2y, (ty + 1)*BSIZE);
	const int n2x_bound = min(N2x, (tx + 1)*BSIZE);

	const int n2y_0 = ty * BSIZE;
	const int n2x_0 = tx * BSIZE;

	float t;

	for (int n2y = n2y_0; n2y < n2y_bound; ++n2y)
	for (int n2x = n2x_0; n2x < n2x_bound; ++n2x) {
		t = 0;

		for (int c1 = 0; c1 < C1; ++c1)
		for (int fy = 0; fy < Fy; ++fy)
		for (int fx = 0; fx < Fx; ++fx) {
			t += I[id(c1, n2x + fx, n2y + fy, C1, N1x, N1y)] * F[f_id(c2, c1, fx, fy, C2, C1, Fx, Fy)];
		}

		O[id(c2, n2x, n2y, C2, N2x, N2y)] = ReLU(t);
	}
}

__global__ void convolution_os_is(int N1y, int N1x, int C1,
	                              int N2y, int N2x, int C2,
	                              int N3y, int N3x, int C3,
	                              int F1y, int F1x,
	                              int F2y, int F2x,
	                              const float *I,
	                              const float *F1,
	                              //   const float *B1,
	                              float *O1,
	                              const float *F2,
	                              // const float *B2,
	                              float *O2) {
	const int ty = blockIdx.x; // < N3y / BSIZE 
	const int tx = blockIdx.y; // < N3x / BSIZE
	const int c3 = threadIdx.x;

	const int n3y_bound = min(N3y, (ty + 1)*BSIZE);
	const int n3x_bound = min(N3x, (tx + 1)*BSIZE);

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
			t += I[id(c1, n2x + f1x, n2y + f1y, C1, N1x, N1y)] * F1[f_id(c2, c1, f1x, f1y, C2, C1, F1x, F1y)];
		}
		
		t = ReLU(t);
		O1[id(c2, n2x, n2y, C2, N2x, N2y)] = t;

		// update values in buffer
		// for (int c3 = 0; c3 < C3; ++c3)
		for (int n3y = max(n3y_0, n2y - F2y + 1); n3y < n3y_bound && n3y <= n2y; ++n3y)
		for (int n3x = max(n3x_0, n2x - F2x + 1); n3x < n3x_bound && n3x <= n2x; ++n3x) {
			int f2y = n2y - n3y, f2x = n2x - n3x;
			buffer[n3y - n3y_0][n3x - n3x_0] += t * F2[f_id(c3, c2, f2x, f2y, C3, C2, F2x, F2y)];
		}
	}

	// write buffer
	for (int i = 0; i < n3y_bound - n3y_0; ++i)
	for (int j = 0; j < n3x_bound - n3x_0; ++j)
		O2[id(c3, n3x_0 + j, n3y_0 + i, C3, N3x, N3y)] = ReLU(buffer[i][j]);
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

// Helper function for using CUDA to add vectors in parallel.
cudaError_t make_two_concolution(float *A,
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


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (check_error_status(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n"))
		goto Error;


	// Allocate buffers on GPU
	cudaStatus = cudaMalloc((void**)&dev_A,  c1*n1*n1 * sizeof(float));
	if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
		goto Error;

	cudaStatus = cudaMalloc((void**)&dev_F1, c2*c1*f1*f1 * sizeof(float));
	if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
		goto Error;

	cudaStatus = cudaMalloc((void**)&dev_C,  c2*n2*n2 * sizeof(float));
	if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
		goto Error;

	cudaStatus = cudaMalloc((void**)&dev_F2, c3*c2*f2*f2 * sizeof(float));
	if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
		goto Error;

	cudaStatus = cudaMalloc((void**)&dev_E,  c3*n3*n3 * sizeof(float));
	if (check_error_status(cudaStatus, "cudaMalloc failed!\n"))
		goto Error;


	// Copy input data from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_A,  A,       c1*n1*n1 * sizeof(float),    cudaMemcpyHostToDevice);
	if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
		goto Error;

	cudaStatus = cudaMemcpy(dev_F1, Filter1, c2*c1*f1*f1 * sizeof(float), cudaMemcpyHostToDevice);
	if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
		goto Error;

	cudaStatus = cudaMemcpy(dev_F2, Filter2, c3*c2*f2*f2 * sizeof(float), cudaMemcpyHostToDevice);
	if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
		goto Error;

	// Run first convolution processing
    cudaEventRecord(startGPU, 0);

	dim3 dimBlock1(c2, 1);
	dim3 dimGrid1((n2 + BSIZE - 1) / BSIZE, (n2 + BSIZE - 1) / BSIZE);
	// Launch a kernel on the GPU with one thread for each element.
	convolution_simple<<<dimGrid1, dimBlock1>>>(n1, n1, c1, 
		                                        n2, n2, c2, 
		                                        f1, f1, 
		                                        dev_A, dev_F1, dev_C);

	//// Check for any errors launching the kernel
	//cudaStatus = cudaGetLastError();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	//	goto Error;
	//}

	//// cudaDeviceSynchronize waits for the kernel to finish, and returns
	//// any errors encountered during the launch.
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	//	goto Error;
	//}

	// Run second convolution processing

	dim3 dimBlock2(c3, 1);
	dim3 dimGrid2((n3 + BSIZE - 1) / BSIZE, (n3 + BSIZE - 1) / BSIZE);
	// Launch a kernel on the GPU with one thread for each element.
	convolution_simple<<<dimGrid2, dimBlock2>>>(n2, n2, c2,
		                                        n3, n3, c3,
		                                        f2, f2,
		                                        dev_C, dev_F2, dev_E);

    cudaEventRecord(stopGPU, 0);
    cudaEventSynchronize(stopGPU);

	//// Check for any errors launching the kernel
	//cudaStatus = cudaGetLastError();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	//	goto Error;
	//}

	//// cudaDeviceSynchronize waits for the kernel to finish, and returns
	//// any errors encountered during the launch.
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	//	goto Error;
	//}


	// Copy output result from GPU
	cudaStatus = cudaMemcpy(C, dev_C, c2*n2*n2 * sizeof(float), cudaMemcpyDeviceToHost);
	if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
		goto Error;

	cudaStatus = cudaMemcpy(E, dev_E, c3*n3*n3 * sizeof(float), cudaMemcpyDeviceToHost);
	if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
		goto Error;

	float elapsedTimeGPU;
	cudaEventElapsedTime(&elapsedTimeGPU, startGPU, stopGPU);
	fprintf(stdout, " %.3f", elapsedTimeGPU);
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

// Helper function for using CUDA to add vectors in parallel.
cudaError_t make_two_concolution_os_is(float *A,
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
    cudaEventRecord(startGPU, 0);

	dim3 dimBlock1(c3, 1);
	dim3 dimGrid1((n3 + BSIZE - 1) / BSIZE, (n3 + BSIZE - 1) / BSIZE);
	// Launch a kernel on the GPU with one thread for each element.
	convolution_os_is<<<dimGrid1, dimBlock1>>>(n1, n1, c1,
		                                       n2, n2, c2,
		                                       n3, n3, c3,
		                                       f1, f1,
		                                       f2, f2,
		                                       dev_A, dev_F1, dev_C, dev_F2, dev_E);

    cudaEventRecord(stopGPU, 0);
    cudaEventSynchronize(stopGPU);

	//// Check for any errors launching the kernel
	//cudaStatus = cudaGetLastError();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	//	goto Error;
	//}

	//// cudaDeviceSynchronize waits for the kernel to finish, and returns
	//// any errors encountered during the launch.
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	//	goto Error;
	//}


	// Copy output result from GPU
	cudaStatus = cudaMemcpy(C, dev_C, c2*n2*n2 * sizeof(float), cudaMemcpyDeviceToHost);
	if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
		goto Error;

	cudaStatus = cudaMemcpy(E, dev_E, c3*n3*n3 * sizeof(float), cudaMemcpyDeviceToHost);
	if (check_error_status(cudaStatus, "cudaMemcpy failed!\n"))
		goto Error;

	float elapsedTimeGPU;
	cudaEventElapsedTime(&elapsedTimeGPU, startGPU, stopGPU);
	fprintf(stdout, " %.3f", elapsedTimeGPU);
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


bool test_convolutions(int N, int F1, int F2, int C1, int C2, int C3) {
	int N1 = N;
	int N2 = N1 - F1 + 1;
	int N3 = N2 - F2 + 1;

	std::cout << N1 << " " << F1 << " " << F2 << " " << C1 << " " << C2 << " " << C3;

	std::vector<float> A(C1*N1*N1);
	std::vector<float> B(C2*C1*F1*F1);
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

	for (int c1 = 0; c1 < C1; c1++)
	for (int c2 = 0; c2 < C2; c2++)
	for (int x = 0; x < F1; x++)
	for (int y = 0; y < F1; y++) {
		B[f_id(c2, c1, x, y, C2, C1, F1, F1)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
	}

	for (int c3 = 0; c3 < C3; c3++)
	for (int c2 = 0; c2 < C2; c2++)
	for (int x = 0; x < F2; x++)
	for (int y = 0; y < F2; y++) {
		D[f_id(c3, c2, x, y, C3, C2, F2, F2)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
	}

	cudaError cudaStatus;

	//std::cout << "Simple convolution" << std::endl;

	cudaStatus = make_two_concolution(A.data(), B.data(), C_1.data(),
		                              D.data(), E_1.data(),
		                              N1, C1, N2, C2, N3, C3, F1, F2);

	if (check_error_status(cudaStatus, "Simple convolution failed!\n"))
		return false;

	//std::cout << "OS-IS convolution" << std::endl;

	cudaStatus = make_two_concolution_os_is(A.data(), B.data(), C_2.data(),
				                            D.data(), E_2.data(),
				                            N1, C1, N2, C2, N3, C3, F1, F2);

	if (check_error_status(cudaStatus, "OS-IS convolution failed!\n"))
		return false;

    std::cout << std::endl;

	bool is_Passed = true;

	//is_Passed &= compare_convolution(N2, N2, F1, F1, C1, C2, A, B, C_1, 1e-1);
	//is_Passed &= compare_convolution(N3, N3, F2, F2, C2, C3, C_1, D, E_1, 1e-1);

	//is_Passed &= compare_results(E_1, E_2, 1e-1);

	return is_Passed;
}


int main()
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(0);
    freopen("parameters.csv", "r", stdin);
    freopen("results.csv", "w", stdout);

    //bool is_Passed = test_convolutions(1, 100, 3, 64, 64);
    bool is_Passed = true;

    int N, F1, F2, C1, C2, C3;

    while (std::cin >> N >> F1 >> F2 >> C1 >> C2 >> C3) {
        is_Passed &= test_convolutions(N, F1, F2, C1, C2, C3);
    }

    /*std::cout << "Total: ";
    if (is_Passed) {
        std::cout << "Passed!" << std::endl;
    }
    else {
        std::cout << "Failed!" << std::endl;
    }*/

    return 0;
}
