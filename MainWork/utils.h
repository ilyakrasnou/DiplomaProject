#include <vector>
#include <iostream>

float* read_image(const char *filename,
                  int* widthOut,
                  int* heightOut);

void store_image(float *imageOut,
                 const char *filename,
                 int cols,
                 const char* refFilename);

char* read_kernel_from_file(const char* kernelPath);

bool read_kernel_binary(const char* filename,
                        uint8_t** data,
                        size_t* size);

bool float_compare(float lhs,
                   float rhs,
                   float eps);

template<typename T>
void print_matrix(const std::vector<T>& matrix,
                  int n,
                  int m) {
    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < m; ++j) {
            std::cout << matrix[i * m + j] << (j == m - 1 ? '\n' : ' ');
        }
    }
}

template<typename T>
bool test_convolution(int n, int m, int n1, int m1,
                      const std::vector<T>& A,
                      const std::vector<T>& Filter,
                      const std::vector<T>& C,
                      float eps = 1e-7) {

    bool isPassed = true;
    for (size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < m; ++j) {
            float val = 0;
            int x = i - n1 / 2;

            for(size_t i1 = 0; i1 < n1; ++i1, ++x) {
                int y = j - m1 / 2;
                for(size_t j1 = 0; j1 < m1; ++j1, ++y) {
                    if(x >= 0 && y >= 0 && x < n && y < m) {
                        val += (Filter[i1 * m1 + j1] * A[x * m + y]);
                    }
                }
            }

            isPassed &= float_compare(val, C[i * m + j], eps);
        }
    }

    if(isPassed) {
        std::cout << "Passed!" << std::endl;
    }
    else {
        std::cout << "Failed!" << std::endl;
    }

    return isPassed;
}

template<typename T>
bool test_max_pool(int nc, int mc, int n1, int m1,
                   const std::vector<T>& A,
                   const std::vector<T>& C,
                   float eps = 1e-7) {

    bool isPassed = true;
    for (size_t i = 0; i < n1; ++i) {
        for(size_t j = 0; j < m1; ++j) {
            float a1 = -1e9,a2 = -1e9,a3 = -1e9,a4 = -1e9;
            if(i * 2 * mc + j * 2 < mc * nc) {
                a1 = A[i * 2 * mc + j * 2];
            }
            if(i * 2 * mc + j * 2 + 1 < mc * nc) {
                a2 = A[i * 2 * mc + j * 2 + 1];
            }
            if((i * 2 + 1) * mc + j * 2 < mc * nc) {
                a3 = A[(i * 2 + 1) * mc + j * 2];
            }
            if((i * 2 + 1) * mc + (j * 2 + 1) < mc * nc) {
                a4 = A[(i * 2 + 1) * mc + (j * 2 + 1)];
            }

            a1 = fmax(a1, a2);
            a3 = fmax(a3, a4);
            a1 = fmax(a1, a3);

            isPassed &= float_compare(C[i * m1 + j], a1, eps);
        }
    }

    if(isPassed) {
        std::cout << "Passed!" << std::endl;
    }
    else {
        std::cout << "Failed!" << std::endl;
    }

    return isPassed;
}

template<typename T>
bool test_matrix_mul(int n, int m, int k,
                     const std::vector<T>& A,
                     const std::vector<T>& B,
                     const std::vector<T>& C,
                     float eps = 1e-10) {
    bool isPassed = true;
    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < m; ++j) {
            float val = 0.0f;
            for(size_t p = 0; p < k; ++p) {
                val += A[i * k + p] * B[p * m + j];
            }
        //    if(!float_compare(val, C[i * m + j], eps)) {
        //        std::cout << "Error " << val << " " << C[i * m + j] << std::endl;
        //    }
            isPassed &= float_compare(val, C[i * m + j], eps);
        }
    }

    if(isPassed) {
        std::cout << "Passed!" << std::endl;
    }
    else {
        std::cout << "Failed!" << std::endl;
    }

    return isPassed;
}

// template<typename T>
// bool test_two_matrix_mul(int n, int m, int k, int q,
//                          const std::vector<T>& A,
//                          const std::vector<T>& B,
//                          const std::vector<T>& C,
//                          const std::vector<T>& X,
//                          const std::vector<T>& Y,
//                          float eps = 1e-7) {
//     bool isPassed = true;
//     for(size_t i = 0; i < n; ++i) {
//         for(size_t j = 0; j < m; ++j) {
//             float val = 0.0f;
//             for(size_t p = 0; p < k; ++p) {
//                 val += A[i * k + p] * B[p * m + j];
//             }
//             if(!float_compare(val, X[i * m + j], eps)) {
//                 std::cout << "Fuck " << val << " " << X[i * m + j] << std::endl;
//             }
//             isPassed &= float_compare(val, X[i * m + j], eps);
//         }
//     }

//     if(!isPassed) {
//         std::cout << "Failed!" << std::endl;
//         return isPassed;
//     }

//     for(size_t i = 0; i < n; ++i) {
//         for(size_t j = 0; j < q; ++j) {
//             float val = 0.0f;
//             for(size_t p = 0; p < m; ++p) {
//                 val += X[i * m + p] * C[p * q + j];
//             }
//             if(!float_compare(val, Y[i * q + j], eps)) {
//                 std::cout << "Fuck 2 " << val << " " << Y[i * q + j] << std::endl;
//             }
//             isPassed &= float_compare(val, Y[i * q + j], eps);
//         }
//     }

//     if(isPassed) {
//         std::cout << "Passed!" << std::endl;
//     }
//     else {
//         std::cout << "Failed!" << std::endl;
//     }

//     return isPassed;
// }

template<typename T>
T find_divisor(T n) {
    T fin;
    for(T i = 2; i * i <= n; ++i) {
        if(n % i == 0) {
            return i;
        }
    }

    return n;
}