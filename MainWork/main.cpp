#include "connected_libs.h"

#include "utils.h"

double time_taken = 0.0;
double eps = 1e-7;

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     break;                                                            \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                     \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       break;                                                          \
     }                                                                 \
     _ret;                                                             \
   })

//OpenCl
struct CLVars {
    cl_platform_id *platforms = nullptr;
    cl_uint num_platforms;
    cl_int clStatus;
    cl_device_id *device_list = nullptr;
    cl_uint num_devices;
    cl_context context = nullptr;
    cl_kernel kernel = nullptr;
    cl_command_queue command_queue = nullptr;
    cl_program program = nullptr;
    char *kernel_string = nullptr;

    //vortex
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
};
//

void cl_clean(CLVars& cl_vars) {
    if(cl_vars.device != nullptr) {
        clReleaseDevice(cl_vars.device);
    }
    if (cl_vars.command_queue != nullptr) {
        clReleaseCommandQueue(cl_vars.command_queue);
    }
    if (cl_vars.kernel != nullptr) {
        clReleaseKernel(cl_vars.kernel);
    }
    if (cl_vars.program != nullptr) {
        clReleaseProgram(cl_vars.program);
    }
    if (cl_vars.context != nullptr) {
        clReleaseContext(cl_vars.context);
    }
    if(cl_vars.platforms != nullptr) {
        free(cl_vars.platforms);
    }
    if(cl_vars.device_list != nullptr) {
        free(cl_vars.device_list);
    }
}

void opencl_environment_definition_vortex(CLVars& cl_vars,
                                   const char* binary_source) {

    //currently in work!!!
    uint8_t *kernel_bin = NULL;
    size_t kernel_size;

    clGetPlatformIDs(1, &cl_vars.platform, NULL);
    clGetDeviceIDs(cl_vars.platform, CL_DEVICE_TYPE_DEFAULT, 1, &cl_vars.device, NULL);
    cl_vars.context = clCreateContext(NULL, 1, &cl_vars.device, NULL, NULL, &cl_vars.clStatus);

    if (read_kernel_binary(binary_source, &kernel_bin, &kernel_size) == false) {
        return;
    }

    cl_vars.program = clCreateProgramWithBinary(cl_vars.context, 1, &cl_vars.device, &kernel_size,
                                                (const uint8_t**)&kernel_bin, &cl_vars.clStatus, NULL);
    if (cl_vars.program == NULL) {
        printf("Binary file load failed!");
        return;
    }
    clBuildProgram(cl_vars.program, 1, &cl_vars.device, NULL, NULL, NULL);
    cl_vars.command_queue = clCreateCommandQueue(cl_vars.context, cl_vars.device, 0, &cl_vars.clStatus);
}

void opencl_environment_definition(CLVars& cl_vars,
                                   const char* kernel_source) {
    clGetPlatformIDs(0, NULL, &cl_vars.num_platforms);
    cl_vars.platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * cl_vars.num_platforms);
    clGetPlatformIDs(cl_vars.num_platforms, cl_vars.platforms, NULL);
    clGetDeviceIDs(cl_vars.platforms[1], CL_DEVICE_TYPE_GPU, 0, NULL, &cl_vars.num_devices);
    cl_vars.device_list = (cl_device_id *) malloc(sizeof(cl_device_id) * cl_vars.num_devices);
    clGetDeviceIDs(cl_vars.platforms[1], CL_DEVICE_TYPE_GPU, cl_vars.num_devices, cl_vars.device_list, NULL);
    cl_vars.context = clCreateContext(NULL, cl_vars.num_devices, cl_vars.device_list, NULL, NULL, &cl_vars.clStatus);
    cl_vars.command_queue = clCreateCommandQueue(cl_vars.context, cl_vars.device_list[0], 0, &cl_vars.clStatus);

    // print device name
    // char deviceName[1024];
    // clGetDeviceInfo(cl_vars.device_list[0], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    // printf("Device name %s \n", deviceName);

    if(cl_vars.kernel_string == nullptr) {
        cl_vars.kernel_string = read_kernel_from_file(kernel_source);
    }
    const char* cKernel_string = cl_vars.kernel_string;

    cl_vars.program = clCreateProgramWithSource(cl_vars.context, 1, &cKernel_string, NULL, &cl_vars.clStatus);
}

void opencl_create_program_conv(CLVars& cl_vars,
                                const char* kernel_name,
                                float *A,
                                float *Filter,
                                float *C,
                                int n, int m, int n1, int m1)   {
    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, n * m * sizeof(float),
                                    NULL, &cl_vars.clStatus);
    cl_mem Filter_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, n1 * m1 * sizeof(float),
                                         NULL, &cl_vars.clStatus);
    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY, n * m * sizeof(float), NULL,
                                    &cl_vars.clStatus);

    clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                                            n * m * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl_vars.command_queue, Filter_clmem, CL_TRUE, 0,
                                            n1 * m1 * sizeof(float), Filter, 0, NULL, NULL);

    clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL);

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &n);
    clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &m);
    clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &n1);
    clSetKernelArg(cl_vars.kernel, 3, sizeof(int), (void *) &m1);
    clSetKernelArg(cl_vars.kernel, 4, sizeof(cl_mem), (void *) &A_clmem);
    clSetKernelArg(cl_vars.kernel, 5, sizeof(cl_mem), (void *) &Filter_clmem);
    clSetKernelArg(cl_vars.kernel, 6, sizeof(cl_mem), (void *) &C_clmem);

    size_t global_size[2];
    global_size[0] = n;
    global_size[1] = m;

    clock_t t;
    t = clock();

    cl_vars.clStatus |= clEnqueueNDRangeKernel(cl_vars.command_queue, cl_vars.kernel, 2, NULL,
                                              global_size, NULL, 0, NULL, NULL);

    cl_vars.clStatus |= clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                           n * m * sizeof(float), C, 0, NULL, NULL);

    t = clock() - t;
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    clReleaseMemObject(A_clmem);
    clReleaseMemObject(Filter_clmem);
    clReleaseMemObject(C_clmem);
}

void opencl_create_program_max_pool(CLVars& cl_vars,
                                    const char* kernel_name,
                                    float *A,
                                    float *C,
                                    int n, int m) {
    int nc = n;
    int mc = m;
    n = (n + (n & 1));
    m = (m + (m & 1));

    int n1 = n / 2;
    int m1 = m / 2;

    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    nc * mc * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY,
                                    n1 * m1 * sizeof(float), NULL, &cl_vars.clStatus);

    clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                                            nc * mc * sizeof(float), A, 0, NULL, NULL);

    clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL);

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &n);
    clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &m);
    clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &nc);
    clSetKernelArg(cl_vars.kernel, 3, sizeof(int), (void *) &mc);
    clSetKernelArg(cl_vars.kernel, 4, sizeof(cl_mem), (void *) &A_clmem);
    clSetKernelArg(cl_vars.kernel, 5, sizeof(cl_mem), (void *) &C_clmem);

    size_t global_size[2];
    size_t local_size[2];

    global_size[0] = n;
    global_size[1] = m;
    local_size[0] = 2;
    local_size[1] = 2;

    clock_t t;
    t = clock();

    CL_CHECK(clEnqueueNDRangeKernel(cl_vars.command_queue, cl_vars.kernel, 2, NULL,
                                              global_size, local_size, 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                           n1 * m1 * sizeof(float), C, 0, NULL, NULL));

    t = clock() - t;
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    clReleaseMemObject(A_clmem);
    clReleaseMemObject(C_clmem);
}

void opencl_create_program_matrix_mul(CLVars& cl_vars,
                                      const char* kernel_name,
                                      float *A,
                                      float *B,
                                      float *C,
                                      int n, int m, int k, int TS) {

    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    n * k * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem B_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    k * m * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_WRITE,
                                    n * m * sizeof(float), NULL, &cl_vars.clStatus);

    clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                                             n * k * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl_vars.command_queue, B_clmem, CL_TRUE, 0,
                                             k * m * sizeof(float), B, 0, NULL, NULL);

    CL_CHECK(clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL));

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    int k1 = k;

    if(k % TS != 0) {
        k += TS - (k % TS);
    }

    std::cout << k << std::endl;

    clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &n);
    clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &m);
    clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &k);
    clSetKernelArg(cl_vars.kernel, 3, sizeof(int), (void *) &k1);
    clSetKernelArg(cl_vars.kernel, 4, sizeof(int), (void *) &TS);
    clSetKernelArg(cl_vars.kernel, 5, sizeof(cl_mem), (void *) &A_clmem);
    clSetKernelArg(cl_vars.kernel, 6, sizeof(cl_mem), (void *) &B_clmem);
    clSetKernelArg(cl_vars.kernel, 7, sizeof(cl_mem), (void *) &C_clmem);
    clSetKernelArg(cl_vars.kernel, 8, TS * TS * sizeof(float), NULL);
    clSetKernelArg(cl_vars.kernel, 9, TS * TS * sizeof(float), NULL);

    size_t global_size[2];
    size_t local_size[2];

    global_size[0] = n;
    global_size[1] = m;

    if(global_size[0] % TS != 0) {
        global_size[0] += TS - (global_size[0] % TS);
    }
    if(global_size[1] % TS != 0) {
        global_size[1] += TS - (global_size[1] % TS);
    }

    local_size[0] = TS;
    local_size[1] = TS;

    clock_t t;
    t = clock();

    CL_CHECK(clEnqueueNDRangeKernel(cl_vars.command_queue, cl_vars.kernel, 2, NULL,
                                               global_size, local_size, 0, NULL, NULL));

    CL_CHECK(clFinish(cl_vars.command_queue));

    clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                            n * m * sizeof(float), C, 0, NULL, NULL);

    t = clock() - t;
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    clReleaseMemObject(A_clmem);
    clReleaseMemObject(B_clmem);
    clReleaseMemObject(C_clmem);
}

std::vector<float> make_matrix_mul(CLVars& cl_vars) {
    opencl_environment_definition(cl_vars, "kernel_matrix_mul.cl");

    // int n = rand() % 500 + 3, m = rand() % 500 + 3, k = rand() % 500 + 3;
    // int n = 65, m = 87, k = 54;
    // int n = 1023, m = 256, k = 1024;
    int n = rand() % 1000 + 3, m = rand() % 1000 + 3, k = rand() % 1000 + 3, TS = 8;

    std::cout << n << " " << m << " " << k << std::endl;

    std::vector<float> A(n * k);
    std::vector<float> B(k * m);
    std::vector<float> C(n * m);

    for (size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < k; ++j) {
            A[i * k + j] = 3.1 * (float)(rand() % 3 + 1);
        }
    }

    for (size_t i = 0; i < k; i++) {
        for(size_t j = 0; j < m; ++j) {
            B[i * m + j] = 3.3 * (float)(rand() % 3 + 1);
        }
    }

    for (size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < m; ++j) {
            C[i * m + j] = 0.0;
        }
    }

    // print_matrix(A, n, k);
    // print_matrix(B, k, m);

    opencl_create_program_matrix_mul(cl_vars, "matrix_mul",
                                     A.data(), B.data(), C.data(), n, m, k, TS);

    // print_matrix(C, n, m);

    assert(test_matrix_mul(n, m, k, A, B, C, eps));

    printf("kernels took %f seconds to execute \n", time_taken);

    time_taken = 0.0f;

    return C;
}

void opencl_create_program_two_matrix_mul(CLVars& cl_vars,
                                          const char* kernel_name,
                                          float *A,
                                          float *B,
                                          float *C,
                                          float *X,
                                          float *Y,
                                          int n, int m, int k, int q) {
    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    n * k * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem B_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    k * m * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    m * q * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem X_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_WRITE,
                                    n * m * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem Y_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_WRITE,
                                    n * q * sizeof(float), NULL, &cl_vars.clStatus);

    clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                                             n * k * sizeof(float), A, 0, NULL, NULL);

    clEnqueueWriteBuffer(cl_vars.command_queue, B_clmem, CL_TRUE, 0,
                                             k * m * sizeof(float), B, 0, NULL, NULL);

    clEnqueueWriteBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                             m * q * sizeof(float), C, 0, NULL, NULL);

    CL_CHECK(clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL));

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    // std::cout << n << " " << m << " "<< k << " " << q << std::endl;

    clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &n);
    clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &m);
    clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &k);
    clSetKernelArg(cl_vars.kernel, 3, sizeof(int), (void *) &q);
    clSetKernelArg(cl_vars.kernel, 4, sizeof(cl_mem), (void *) &A_clmem);
    clSetKernelArg(cl_vars.kernel, 5, sizeof(cl_mem), (void *) &B_clmem);
    clSetKernelArg(cl_vars.kernel, 6, sizeof(cl_mem), (void *) &C_clmem);
    clSetKernelArg(cl_vars.kernel, 7, sizeof(cl_mem), (void *) &X_clmem);
    clSetKernelArg(cl_vars.kernel, 8, sizeof(cl_mem), (void *) &Y_clmem);
    
    size_t global_size[1];
    size_t local_size[1];

    local_size[0] = std::max(m, q);
    global_size[0] = n * local_size[0];

    // std::cout << global_size[0] << " " << local_size[0] << std::endl;

    clock_t t;
    t = clock();

    CL_CHECK(clEnqueueNDRangeKernel(cl_vars.command_queue, cl_vars.kernel, 1, NULL,
                                    global_size, local_size, 0, NULL, NULL));

    CL_CHECK(clFinish(cl_vars.command_queue));

    clEnqueueReadBuffer(cl_vars.command_queue, X_clmem, CL_TRUE, 0,
                        n * m * sizeof(float), X, 0, NULL, NULL);
    clEnqueueReadBuffer(cl_vars.command_queue, Y_clmem, CL_TRUE, 0,
                        n * q * sizeof(float), Y, 0, NULL, NULL);

    t = clock() - t;
    printf("Clear execution time %f miliseconds \n", (double)t);
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    clReleaseMemObject(A_clmem);
    clReleaseMemObject(B_clmem);
    clReleaseMemObject(C_clmem);
    clReleaseMemObject(X_clmem);
    clReleaseMemObject(Y_clmem);
}

void opencl_create_program_two_matrix_mul_os_is(CLVars& cl_vars,
                                                const char* kernel_name,
                                                float *A,
                                                float *B,
                                                float *C,
                                                float *X,
                                                float *Y,
                                                int n, int m, int k, int q) {
    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    n * k * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem B_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    k * m * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    m * q * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem X_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_WRITE,
                                    n * m * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem Y_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_WRITE,
                                    n * q * sizeof(float), NULL, &cl_vars.clStatus);
    // cl_mem block_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_WRITE,
    //                                 q * sizeof(int), NULL, &cl_vars.clStatus);

    clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                                             n * k * sizeof(float), A, 0, NULL, NULL);

    clEnqueueWriteBuffer(cl_vars.command_queue, B_clmem, CL_TRUE, 0,
                                             k * m * sizeof(float), B, 0, NULL, NULL);

    clEnqueueWriteBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                             m * q * sizeof(float), C, 0, NULL, NULL);

    // std::vector<float> block(q, 0);
    // clEnqueueWriteBuffer(cl_vars.command_queue, block_clmem, CL_TRUE, 0,
    //                                          q * sizeof(int), block.data(), 0, NULL, NULL);

    CL_CHECK(clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL));

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    // std::cout << n << " " << m << " "<< k << " " << q << std::endl;

    clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &n);
    clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &m);
    clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &k);
    clSetKernelArg(cl_vars.kernel, 3, sizeof(int), (void *) &q);
    clSetKernelArg(cl_vars.kernel, 4, sizeof(cl_mem), (void *) &A_clmem);
    clSetKernelArg(cl_vars.kernel, 5, sizeof(cl_mem), (void *) &B_clmem);
    clSetKernelArg(cl_vars.kernel, 6, sizeof(cl_mem), (void *) &C_clmem);
    clSetKernelArg(cl_vars.kernel, 7, sizeof(cl_mem), (void *) &X_clmem);
    clSetKernelArg(cl_vars.kernel, 8, sizeof(cl_mem), (void *) &Y_clmem);
    // clSetKernelArg(cl_vars.kernel, 9, 1 * sizeof(int), NULL);
    clSetKernelArg(cl_vars.kernel, 9, q * m * sizeof(float), NULL);
    
    
    size_t global_size[1];
    size_t local_size[1];

    // local_size[0] = m;
    // global_size[0] = n * local_size[0];

    local_size[0] = std::max(m, q);
    global_size[0] = n * local_size[0];

    // std::cout << global_size[0] << " " << local_size[0] << std::endl;

    clock_t t;
    t = clock();

    CL_CHECK(clEnqueueNDRangeKernel(cl_vars.command_queue, cl_vars.kernel, 1, NULL,
                                               global_size, local_size, 0, NULL, NULL));

    CL_CHECK(clFinish(cl_vars.command_queue));

    clEnqueueReadBuffer(cl_vars.command_queue, X_clmem, CL_TRUE, 0,
                                            n * m * sizeof(float), X, 0, NULL, NULL);
    clEnqueueReadBuffer(cl_vars.command_queue, Y_clmem, CL_TRUE, 0,
                                            n * q * sizeof(float), Y, 0, NULL, NULL);

    t = clock() - t;
    printf("Clear execution time %f miliseconds \n", (double)t);
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    clReleaseMemObject(A_clmem);
    clReleaseMemObject(B_clmem);
    clReleaseMemObject(C_clmem);
    clReleaseMemObject(X_clmem);
    clReleaseMemObject(Y_clmem);
}

bool make_two_matrix_mul(CLVars& cl_vars) {
    //opencl_environment_definition_vortex(c    l_vars, "kernel_matrix_mul.pocl");

    int n = rand() % 500 + 3, m = rand() % 100 + 3, k = rand() % 500 + 3, q = rand() % 100 + 3;
    // int n = rand() % 500 + 3, m = 10, k = rand() % 500 + 3, q = rand() % 500 + 3;
    // int n = 65, m = 87, k = 54, q = 74;
    // int n = 4, m = 10, k = 5, q = 3;

    // std::cout << n << " " << m << " " << k << " " << q << std::endl;

    std::vector<float> A(n * k);
    std::vector<float> B(k * m);
    std::vector<float> C(m * q);
    std::vector<float> X(n * m);
    std::vector<float> Y(n * q);

    for (size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < k; ++j) {
            A[i * k + j] = 1 * (float)(rand() % 3 + 1);
        }
    }

    for (size_t i = 0; i < k; i++) {
        for(size_t j = 0; j < m; ++j) {
            B[i * m + j] = 2 * (float)(rand() % 3 + 1);
        }
    }

    for (size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < q; ++j) {
            C[i * q + j] = 3 * (float)(rand() % 3 + 1);
        }
    }

    for (size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < m; ++j) {
            X[i * m + j] = 0.0;
        }
    }

    for (size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < q; ++j) {
            Y[i * q + j] = 0.0;
        }
    }

    // printf("Matrix A: \n");
    // print_matrix(A, n, k);
    // printf("Matrix B: \n");
    // print_matrix(B, k, m);
    // printf("Matrix C: \n");
    // print_matrix(C, m, q);

    bool isPassed = true;

    opencl_environment_definition(cl_vars, "kernel_two_matrix_mul.cl");
    printf("Direct two matrix mul: ");
    opencl_create_program_two_matrix_mul(cl_vars, "two_matrix_mul",
                                         A.data(), B.data(), C.data(), X.data(), Y.data(), 
                                         n, m, k, q);
    
    isPassed &= test_matrix_mul(n, m, k, A, B, X);
    isPassed &= test_matrix_mul(n, q, m, X, C, Y);

    cl_clean(cl_vars);

    // printf("Matrix X: \n");
    // print_matrix(X, n, m);
    // printf("Matrix Y: \n");
    // print_matrix(Y, n, q);

    opencl_environment_definition(cl_vars, "kernel_two_matrix_mul.cl");
    printf("OS->IS two matrix mul: ");
    opencl_create_program_two_matrix_mul_os_is(cl_vars, "two_matrix_mul_os_is",
                                               A.data(), B.data(), C.data(), X.data(), Y.data(), 
                                               n, m, k, q);

    isPassed &= test_matrix_mul(n, m, k, A, B, X);
    isPassed &= test_matrix_mul(n, q, m, X, C, Y);

    cl_clean(cl_vars);

    // printf("Matrix X: \n");
    // print_matrix(X, n, m);
    // printf("Matrix Y: \n");
    // print_matrix(Y, n, q);

    

    // printf("kernels took %f seconds to execute \n", time_taken);

    time_taken = 0.0f;

    return isPassed;
}

std::vector<float> make_convolution(CLVars& cl_vars) {
     opencl_environment_definition(cl_vars, "kernel_conv.cl");

     int n = 1000, m = 1000;
     int n1 = 3, m1 = 3;

     std::vector<float> A(n * m);
     std::vector<float> Filter(n1 * m1);
     std::vector<float> C(n * m);

     for (size_t i = 0; i < n; i++) {
         for(size_t j = 0; j < m; ++j) {
             A[i * m + j] = 2.0 * (rand() % (j + 1));
             C[i * m + j] = 0;
         }
     }

     for (size_t i = 0; i < n1; i++) {
         for(size_t j = 0; j < m1; ++j) {
             Filter[i * m1 + j] = 1.0;
         }
     }

     opencl_create_program_conv(cl_vars, "matrix_convolutional_transformation", A.data(),
                                Filter.data(), C.data(), n, m, n1, m1);

     assert(test_convolution(n, m, n1, m1, A, Filter, C));

     printf("kernels took %f seconds to execute \n", time_taken);

     time_taken = 0.0f;

     return C;
 }

std::vector<float> make_max_pool(CLVars& cl_vars) {
    opencl_environment_definition(cl_vars, "kernel_max_pool.cl");

    int n = rand() % 1000 + 3, m = rand() % 1122 + 3;

    std::cout << n << " " << m << std::endl;

    int n1 = (n + (n & 1)) / 2;
    int m1 = (m + (m & 1)) / 2;

    std::vector<float> A(n * m);
    std::vector<float> C(n1 * m1);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            A[i * m + j] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
        }
    }
    
    //print_matrix(A, n, m);

    opencl_create_program_max_pool(cl_vars, "matrix_max_pool_transformation",
                                   A.data(), C.data(), n, m);
    
    //print_matrix(C, n1, m1);

    assert(test_max_pool(n, m, n1, m1, A, C));

    printf("kernels took %f seconds to execute \n", time_taken);

    time_taken = 0.0f;

    return C;
}

#define BSIZE 16

void opencl_create_program_conv2d(CLVars& cl_vars,
                                  const char* kernel_name,
                                  float *A,
                                  float *Filter,
                                  float *C,
                                  int n1, int c1, int n2, int c2, int f)   {
    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, c1 * n1 * n1 * sizeof(float),
                                    NULL, &cl_vars.clStatus);
    
    if (cl_vars.clStatus != CL_SUCCESS || A_clmem == NULL) {
        std::cout << "OpenCL can't allocate memory for A " << (int) cl_vars.clStatus << std::endl;
        return;
    }

    cl_mem Filter_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, c2 * c1 * f * f * sizeof(float),
                                         NULL, &cl_vars.clStatus);

    if (cl_vars.clStatus != CL_SUCCESS || Filter_clmem == NULL) {
        std::cout << "OpenCL can't allocate memory for Filter " << (int) cl_vars.clStatus << std::endl;
        return;
    }

    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY, c2 * n2 * n2 * sizeof(float), NULL,
                                    &cl_vars.clStatus);

    if (cl_vars.clStatus != CL_SUCCESS || C_clmem == NULL) {
        std::cout << "OpenCL can't allocate memory for C " << (int) cl_vars.clStatus << std::endl;
        return;
    }

    clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                                            c1 * n1 * n1  * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl_vars.command_queue, Filter_clmem, CL_TRUE, 0,
                                            c2 * c1 * f * f * sizeof(float), Filter, 0, NULL, NULL);

    cl_vars.clStatus = clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL);

    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't build programm " << (int) cl_vars.clStatus << std::endl;
        return;
    }

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    cl_vars.clStatus = clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &n1);
    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't pass arguments " << (int) cl_vars.clStatus << std::endl;
        return;
    }
    cl_vars.clStatus = clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &n1);
    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't pass arguments " << (int) cl_vars.clStatus << std::endl;
        return;
    }
    cl_vars.clStatus = clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &c1);
    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't pass arguments " << (int) cl_vars.clStatus << std::endl;
        return;
    }
    cl_vars.clStatus = clSetKernelArg(cl_vars.kernel, 3, sizeof(int), (void *) &n2);
    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't pass arguments " << (int) cl_vars.clStatus << std::endl;
        return;
    }
    cl_vars.clStatus = clSetKernelArg(cl_vars.kernel, 4, sizeof(int), (void *) &n2);
    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't pass arguments " << (int) cl_vars.clStatus << std::endl;
        return;
    }
    cl_vars.clStatus = clSetKernelArg(cl_vars.kernel, 5, sizeof(int), (void *) &c2);
    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't pass arguments " << (int) cl_vars.clStatus << std::endl;
        return;
    }
    cl_vars.clStatus = clSetKernelArg(cl_vars.kernel, 6, sizeof(int), (void *) &f);
    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't pass arguments " << (int) cl_vars.clStatus << std::endl;
        return;
    }
    cl_vars.clStatus = clSetKernelArg(cl_vars.kernel, 7, sizeof(int), (void *) &f);
    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't pass arguments " << (int) cl_vars.clStatus << std::endl;
        return;
    }
    cl_vars.clStatus = clSetKernelArg(cl_vars.kernel, 8, sizeof(cl_mem), (void *) &A_clmem);
    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't pass arguments " << (int) cl_vars.clStatus << std::endl;
        return;
    }
    cl_vars.clStatus = clSetKernelArg(cl_vars.kernel, 9, sizeof(cl_mem), (void *) &Filter_clmem);
    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't pass arguments " << (int) cl_vars.clStatus << std::endl;
        return;
    }
    cl_vars.clStatus = clSetKernelArg(cl_vars.kernel, 10, sizeof(cl_mem), (void *) &C_clmem);

    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't pass arguments " << (int) cl_vars.clStatus << std::endl;
        return;
    }

    size_t global_size1[2];
    size_t local_size1[2];

    local_size1[0] = c2;
    // local_size1[0] = 1;
    local_size1[1] = 1;
    global_size1[0] = ((n2 + BSIZE - 1) / BSIZE) * local_size1[0];
    global_size1[1] = ((n2 + BSIZE - 1) / BSIZE) * local_size1[1];
    // global_size1[0] = n2 * local_size1[0];
    // global_size1[1] = n2 * local_size1[1];

    std::cout << global_size1[0] << " " << global_size1[1] << std::endl;

    clock_t t;
    t = clock();

    cl_vars.clStatus = clEnqueueNDRangeKernel(cl_vars.command_queue, cl_vars.kernel, 2, NULL,
                                              global_size1, local_size1, 0, NULL, NULL);

    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't enqueue kernel " << (int) cl_vars.clStatus << std::endl;
        return;
    }

    cl_vars.clStatus = clFinish(cl_vars.command_queue);

    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't finish queue " << (int) cl_vars.clStatus << std::endl;
        return;
    }

    cl_vars.clStatus = clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                           c2 * n2 * n2 * sizeof(float), C, 0, NULL, NULL);

    if (cl_vars.clStatus != CL_SUCCESS) {
        std::cout << "OpenCL can't read from queue " << (int) cl_vars.clStatus << std::endl;
        return;
    }

    t = clock() - t;
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    clReleaseMemObject(A_clmem);
    clReleaseMemObject(Filter_clmem);
    clReleaseMemObject(C_clmem);
}

bool make_conv2D(CLVars& cl_vars, int i) {
    opencl_environment_definition(cl_vars, "kernel_conv2d.cl");

    int C1 = 64, C2 = 64, F = 3;

    int N1 = i + F;

    std::cout << "Start" << std::endl;

    int N2 = N1 - F + 1;

    std::vector<float> A(C1*N1*N1);
    std::vector<float> B(C2*C1*F*F);
    std::vector<float> C(C2*N2*N2);

    for (int c = 0; c < C1; c++)
    for (int x = 0; x < N1; x++)
    for (int y = 0; y < N1; y++) {
        A[id(c, x, y, C1, N1, N1)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
    }

    for (int c1 = 0; c1 < C1; c1++)
    for (int c2 = 0; c2 < C2; c2++)
    for (int x = 0; x < F; x++)
    for (int y = 0; y < F; y++) {
        B[f_id(c2, c1, x, y, C2, C1, F, F)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
    }

    std::cout << "Initialized" << std::endl;
    //print_matrix(A, n, m);

    opencl_create_program_conv2d(cl_vars, "conv2D_tranform",
                                 A.data(), B.data(), C.data(), N1, C1, N2, C2, F);
    
    //print_matrix(C, n1, m1);
    std::cout << "Finished" << std::endl;

    bool isPassed = test_conv2D(N2, N2, F, F, C1, C2, A, B, C);
    // assert(test_max_pool(n, m, n1, m1, A, C));

    printf("kernels took %f seconds to execute \n", time_taken);

    time_taken = 0.0f;

    return isPassed;
}

void opencl_create_program_two_conv2d(CLVars& cl_vars,
                                      const char* kernel_name,
                                      float *A,
                                      float *Filter1,
                                      float *C,
                                      float *Filter2,
                                      float *E,
                                      int n1, int c1, 
                                      int n2, int c2, 
                                      int n3, int c3, 
                                      int f1, int f2) {
    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, c1 * n1 * n1 * sizeof(float),
                                    NULL, &cl_vars.clStatus);
    cl_mem Filter1_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, c2 * c1 * f1 * f1 * sizeof(float),
                                          NULL, &cl_vars.clStatus);
    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_WRITE, c2 * n2 * n2 * sizeof(float), NULL,
                                    &cl_vars.clStatus);
    cl_mem Filter2_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, c3 * c2 * f2 * f2 * sizeof(float),
                                          NULL, &cl_vars.clStatus);
    cl_mem E_clmem = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY, c3 * n3 * n3 * sizeof(float), NULL,
                                    &cl_vars.clStatus);

    clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                         c1 * n1 * n1  * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl_vars.command_queue, Filter1_clmem, CL_TRUE, 0,
                         c2 * c1 * f1 * f1 * sizeof(float), Filter1, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl_vars.command_queue, Filter2_clmem, CL_TRUE, 0,
                         c3 * c2 * f2 * f2 * sizeof(float), Filter2, 0, NULL, NULL);

    CL_CHECK(clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL));

    cl_kernel kernel1 = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    clSetKernelArg(kernel1, 0, sizeof(int), (void *) &n1);
    clSetKernelArg(kernel1, 1, sizeof(int), (void *) &n1);
    clSetKernelArg(kernel1, 2, sizeof(int), (void *) &c1);
    clSetKernelArg(kernel1, 3, sizeof(int), (void *) &n2);
    clSetKernelArg(kernel1, 4, sizeof(int), (void *) &n2);
    clSetKernelArg(kernel1, 5, sizeof(int), (void *) &c2);
    clSetKernelArg(kernel1, 6, sizeof(int), (void *) &f1);
    clSetKernelArg(kernel1, 7, sizeof(int), (void *) &f1);
    clSetKernelArg(kernel1, 8, sizeof(cl_mem), (void *) &A_clmem);
    clSetKernelArg(kernel1, 9, sizeof(cl_mem), (void *) &Filter1_clmem);
    clSetKernelArg(kernel1, 10, sizeof(cl_mem), (void *) &C_clmem);

    cl_kernel kernel2 = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    clSetKernelArg(kernel2, 0, sizeof(int), (void *) &n2);
    clSetKernelArg(kernel2, 1, sizeof(int), (void *) &n2);
    clSetKernelArg(kernel2, 2, sizeof(int), (void *) &c2);
    clSetKernelArg(kernel2, 3, sizeof(int), (void *) &n3);
    clSetKernelArg(kernel2, 4, sizeof(int), (void *) &n3);
    clSetKernelArg(kernel2, 5, sizeof(int), (void *) &c3);
    clSetKernelArg(kernel2, 6, sizeof(int), (void *) &f2);
    clSetKernelArg(kernel2, 7, sizeof(int), (void *) &f2);
    clSetKernelArg(kernel2, 8, sizeof(cl_mem), (void *) &C_clmem);
    clSetKernelArg(kernel2, 9, sizeof(cl_mem), (void *) &Filter2_clmem);
    clSetKernelArg(kernel2, 10, sizeof(cl_mem), (void *) &E_clmem);

    size_t global_size1[2];
    size_t local_size1[2];

    local_size1[0] = c2;
    // local_size1[0] = 1;
    local_size1[1] = 1;
    global_size1[0] = n2 * local_size1[0];
    global_size1[1] = n2 * local_size1[1];
    
    // global_size1[0] = ((n2 + BSIZE - 1) / BSIZE) * local_size1[0];
    // global_size1[1] = ((n2 + BSIZE - 1) / BSIZE) * local_size1[1];

    std::cout << "1 " << n2 << " " << local_size1[0] << " " << global_size1[0] << std::endl;

    size_t global_size2[2];
    size_t local_size2[2];

    local_size2[0] = c3;
    // local_size2[0] = 1;
    local_size2[1] = 1;
    global_size2[0] = n3 * local_size2[0];
    global_size2[1] = n3 * local_size2[1];
    // global_size2[0] = ((n3 + BSIZE - 1) / BSIZE) * local_size2[0];
    // global_size2[1] = ((n3 + BSIZE - 1) / BSIZE) * local_size2[1];

    std::cout << "2 " << n3 << " " << local_size2[0] << " " << global_size2[0] << std::endl;

    clock_t t;
    t = clock();

    CL_CHECK(clEnqueueNDRangeKernel(cl_vars.command_queue, kernel1, 2, NULL,
                                    global_size1, local_size1, 0, NULL, NULL));

    // CL_CHECK(clFinish(cl_vars.command_queue));

    CL_CHECK(clEnqueueNDRangeKernel(cl_vars.command_queue, kernel2, 2, NULL,
                                    global_size2, local_size2, 0, NULL, NULL));

    CL_CHECK(clFinish(cl_vars.command_queue));

    CL_CHECK(clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                 c2 * n2 * n2 * sizeof(float), C, 0, NULL, NULL));
    
    CL_CHECK(clEnqueueReadBuffer(cl_vars.command_queue, E_clmem, CL_TRUE, 0,
                                 c3 * n3 * n3 * sizeof(float), E, 0, NULL, NULL));

    t = clock() - t;
    printf("Clear execution time %f miliseconds \n", (double)t);
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    clReleaseMemObject(A_clmem);
    clReleaseMemObject(Filter1_clmem);
    clReleaseMemObject(C_clmem);
    clReleaseMemObject(Filter2_clmem);
    clReleaseMemObject(E_clmem);
}

void opencl_create_program_two_conv2d_os_is(CLVars& cl_vars,
                                            const char* kernel_name,
                                            float *A,
                                            float *Filter1,
                                            float *C,
                                            float *Filter2,
                                            float *E,
                                            int n1, int c1, 
                                            int n2, int c2, 
                                            int n3, int c3, 
                                            int f1, int f2) {
    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, c1 * n1 * n1 * sizeof(float),
                                    NULL, &cl_vars.clStatus);
    cl_mem Filter1_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, c2 * c1 * f1 * f1 * sizeof(float),
                                          NULL, &cl_vars.clStatus);
    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_WRITE, c2 * n2 * n2 * sizeof(float), NULL,
                                    &cl_vars.clStatus);
    cl_mem Filter2_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, c3 * c2 * f2 * f2 * sizeof(float),
                                          NULL, &cl_vars.clStatus);
    cl_mem E_clmem = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY, c3 * n3 * n3 * sizeof(float), NULL,
                                    &cl_vars.clStatus);

    clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                         c1 * n1 * n1  * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl_vars.command_queue, Filter1_clmem, CL_TRUE, 0,
                         c2 * c1 * f1 * f1 * sizeof(float), Filter1, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl_vars.command_queue, Filter2_clmem, CL_TRUE, 0,
                         c3 * c2 * f2 * f2 * sizeof(float), Filter2, 0, NULL, NULL);

    // CL_CHECK(clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL));

    cl_kernel kernel1 = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    clSetKernelArg(kernel1, 0, sizeof(int), (void *) &n1);
    clSetKernelArg(kernel1, 1, sizeof(int), (void *) &n1);
    clSetKernelArg(kernel1, 2, sizeof(int), (void *) &c1);
    clSetKernelArg(kernel1, 3, sizeof(int), (void *) &n2);
    clSetKernelArg(kernel1, 4, sizeof(int), (void *) &n2);
    clSetKernelArg(kernel1, 5, sizeof(int), (void *) &c2);
    clSetKernelArg(kernel1, 6, sizeof(int), (void *) &n3);
    clSetKernelArg(kernel1, 7, sizeof(int), (void *) &n3);
    clSetKernelArg(kernel1, 8, sizeof(int), (void *) &c3);
    clSetKernelArg(kernel1, 9, sizeof(int), (void *) &f1);
    clSetKernelArg(kernel1, 10, sizeof(int), (void *) &f1);
    clSetKernelArg(kernel1, 11, sizeof(int), (void *) &f2);
    clSetKernelArg(kernel1, 12, sizeof(int), (void *) &f2);
    clSetKernelArg(kernel1, 13, sizeof(cl_mem), (void *) &A_clmem);
    clSetKernelArg(kernel1, 14, sizeof(cl_mem), (void *) &Filter1_clmem);
    clSetKernelArg(kernel1, 15, sizeof(cl_mem), (void *) &C_clmem);
    clSetKernelArg(kernel1, 16, sizeof(cl_mem), (void *) &Filter2_clmem);
    clSetKernelArg(kernel1, 17, sizeof(cl_mem), (void *) &E_clmem);

    size_t global_size1[2];
    size_t local_size1[2];

    // local_size1[0] = c3;
    // local_size1[1] = 1;
    // global_size1[0] = n3 * local_size1[0];
    // global_size1[1] = n3 * local_size1[1];

    // local_size1[0] = c3;
    local_size1[0] = std::max(c2, c3);
    local_size1[1] = 1;
    global_size1[0] = ((n3 + BSIZE - 1) / BSIZE)  * local_size1[0];
    global_size1[1] = ((n3 + BSIZE - 1) / BSIZE) * local_size1[1];

    // local_size1[0] = (n3 + 31) >> 5;
    // global_size1[0] = ((n3 + 31) >> 5) * local_size1[0];

    std::cout << n3 << " " << local_size1[0] << " " << global_size1[0] << std::endl;

    clock_t t;
    t = clock();

    CL_CHECK(clEnqueueNDRangeKernel(cl_vars.command_queue, kernel1, 2, NULL,
                                    global_size1, local_size1, 0, NULL, NULL));

    CL_CHECK(clFinish(cl_vars.command_queue));

    CL_CHECK(clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                 c2 * n2 * n2 * sizeof(float), C, 0, NULL, NULL));
    
    CL_CHECK(clEnqueueReadBuffer(cl_vars.command_queue, E_clmem, CL_TRUE, 0,
                                 c3 * n3 * n3 * sizeof(float), E, 0, NULL, NULL));

    t = clock() - t;
    printf("Clear execution time %f miliseconds \n", (double)t);
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    clReleaseMemObject(A_clmem);
    clReleaseMemObject(Filter1_clmem);
    clReleaseMemObject(C_clmem);
    clReleaseMemObject(Filter2_clmem);
    clReleaseMemObject(E_clmem);
}

void opencl_create_program_two_conv2d_fusion(CLVars& cl_vars,
                                             const char* kernel_name,
                                             float *A,
                                             float *Filter1,
                                             float *C,
                                             float *Filter2,
                                             float *E,
                                             int n1, int c1, 
                                             int n2, int c2, 
                                             int n3, int c3, 
                                             int f1, int f2) {
    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, c1 * n1 * n1 * sizeof(float),
                                    NULL, &cl_vars.clStatus);
    cl_mem Filter1_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, c2 * c1 * f1 * f1 * sizeof(float),
                                          NULL, &cl_vars.clStatus);
    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_WRITE, c2 * n2 * n2 * sizeof(float), NULL,
                                    &cl_vars.clStatus);
    cl_mem Filter2_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, c3 * c2 * f2 * f2 * sizeof(float),
                                          NULL, &cl_vars.clStatus);
    cl_mem E_clmem = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY, c3 * n3 * n3 * sizeof(float), NULL,
                                    &cl_vars.clStatus);

    clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                         c1 * n1 * n1  * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl_vars.command_queue, Filter1_clmem, CL_TRUE, 0,
                         c2 * c1 * f1 * f1 * sizeof(float), Filter1, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl_vars.command_queue, Filter2_clmem, CL_TRUE, 0,
                         c3 * c2 * f2 * f2 * sizeof(float), Filter2, 0, NULL, NULL);

    int size = BSIZE - f2 + 1;

    // CL_CHECK(clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL));

    cl_kernel kernel1 = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    clSetKernelArg(kernel1, 0, sizeof(int), (void *) &n1);
    clSetKernelArg(kernel1, 1, sizeof(int), (void *) &n1);
    clSetKernelArg(kernel1, 2, sizeof(int), (void *) &c1);
    clSetKernelArg(kernel1, 3, sizeof(int), (void *) &n2);
    clSetKernelArg(kernel1, 4, sizeof(int), (void *) &n2);
    clSetKernelArg(kernel1, 5, sizeof(int), (void *) &c2);
    clSetKernelArg(kernel1, 6, sizeof(int), (void *) &n3);
    clSetKernelArg(kernel1, 7, sizeof(int), (void *) &n3);
    clSetKernelArg(kernel1, 8, sizeof(int), (void *) &c3);
    clSetKernelArg(kernel1, 9, sizeof(int), (void *) &f1);
    clSetKernelArg(kernel1, 10, sizeof(int), (void *) &f1);
    clSetKernelArg(kernel1, 11, sizeof(int), (void *) &f2);
    clSetKernelArg(kernel1, 12, sizeof(int), (void *) &f2);
    clSetKernelArg(kernel1, 13, sizeof(int), (void *) &size);
    clSetKernelArg(kernel1, 14, sizeof(int), (void *) &size);
    clSetKernelArg(kernel1, 15, sizeof(cl_mem), (void *) &A_clmem);
    clSetKernelArg(kernel1, 16, sizeof(cl_mem), (void *) &Filter1_clmem);
    clSetKernelArg(kernel1, 17, sizeof(cl_mem), (void *) &C_clmem);
    clSetKernelArg(kernel1, 18, sizeof(cl_mem), (void *) &Filter2_clmem);
    clSetKernelArg(kernel1, 19, sizeof(cl_mem), (void *) &E_clmem);

    size_t global_size1[2];
    size_t local_size1[2];

    // local_size1[0] = 1;
    local_size1[0] = c3;
    local_size1[1] = 1;

    global_size1[0] = ((n2 + size - 1) / size)  * local_size1[0];
    global_size1[1] = ((n2 + size - 1) / size) * local_size1[1];

    std::cout << n3 << " " << size << " " << local_size1[0] << " " << global_size1[0] << std::endl;

    clock_t t;
    t = clock();

    CL_CHECK(clEnqueueNDRangeKernel(cl_vars.command_queue, kernel1, 2, NULL,
                                    global_size1, local_size1, 0, NULL, NULL));

    CL_CHECK(clFinish(cl_vars.command_queue));

    CL_CHECK(clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                 c2 * n2 * n2 * sizeof(float), C, 0, NULL, NULL));
    
    CL_CHECK(clEnqueueReadBuffer(cl_vars.command_queue, E_clmem, CL_TRUE, 0,
                                 c3 * n3 * n3 * sizeof(float), E, 0, NULL, NULL));

    t = clock() - t;
    printf("Clear execution time %f miliseconds \n", (double)t);
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    clReleaseMemObject(A_clmem);
    clReleaseMemObject(Filter1_clmem);
    clReleaseMemObject(C_clmem);
    clReleaseMemObject(Filter2_clmem);
    clReleaseMemObject(E_clmem);
}


bool make_two_conv2D(CLVars& cl_vars) {
    opencl_environment_definition(cl_vars, "kernel_conv2d.cl");

    int C1 = 1, C2 = 64, C3 = 64, F1 = 3, F2 = 3;
    // int C1 = 1, C2 = 1, C3 = 1, F1 = 3, F2 = 3;
    // int N1 = rand() % 100 + F1 + F2 + 400;
    int N1 = 300 + F1 + F2;
    // int N1 = rand() % 350 + F1 + F2;
    // int N1 = rand() % 200 + 3, C1 = 1, C2 = 64, C3 = 64, F1 = 3, F2 = 3;
    std::cout << "Start" << std::endl;

    int N2 = N1 - F1 + 1;
    int N3 = N2 - F2 + 1;

    std::cout << N1 << " " << N2 << " " << N3 << std::endl;

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

    std::cout << "Initialized" << std::endl;
    //print_matrix(A, n, m);

    opencl_create_program_two_conv2d(cl_vars, "conv2D_tranform",
                                     A.data(), B.data(), C_1.data(), 
                                     D.data(), E_1.data(), 
                                     N1, C1, N2, C2, N3, C3, F1, F2);

    
    opencl_create_program_two_conv2d_fusion(cl_vars, "two_conv2D_fusion",
                                            A.data(), B.data(), C_2.data(), 
                                            D.data(), E_2.data(), 
                                            N1, C1, N2, C2, N3, C3, F1, F2);
    
    // opencl_create_program_two_conv2d_os_is(cl_vars, "two_conv2D_tranform_buffer",
    //                                        A.data(), B.data(), C_2.data(), 
    //                                        D.data(), E_2.data(), 
    //                                        N1, C1, N2, C2, N3, C3, F1, F2);

    //print_matrix(C, n1, m1);
    std::cout << "Finished" << std::endl;
    // assert(test_max_pool(n, m, n1, m1, A, C));

    printf("kernels took %f seconds to execute \n", time_taken);
    time_taken = 0.0f;

    bool isPassed = true;

    // isPassed &= test_conv2D(N2, N2, F1, F1, C1, C2, A, B, C_1, 1e-1);
    // isPassed &= test_conv2D(N3, N3, F2, F2, C2, C3, C_1, D, E_1, 1e-1);

    isPassed &= test_result(E_1, E_2, 1e-1);

    return isPassed;
}

int main (int argc, char **argv) {

    srand(time(nullptr));

    bool isPassed = true;
    CLVars cl_vars;

    for(int i = 100; i < 200; i += 100) {
        isPassed &= make_two_conv2D(cl_vars);
        // cl_clean(cl_vars);
    }

    free(cl_vars.kernel_string);

    std::cout << "Total: ";
    if(isPassed) {
        std::cout << "Passed!" << std::endl;
    }
    else {
        std::cout << "Failed!" << std::endl;
    }

    return 0;
}
