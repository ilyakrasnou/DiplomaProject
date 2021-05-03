#include <vector>
#include <iostream>
#include <optional>

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
bool test_result(const std::vector<T>& A,
                 const std::vector<T>& B,
                 float eps = 1e-7) {
    if (A.size() != B.size())
        return false;

    for (int i = 0; i < A.size(); i++)  {
        // std::cout << i << " " << A[i] << " " << B[i] << std::endl;
        if (!float_compare(A[i], B[i], eps)) {
            std::cout << i << " " << A[i] << " " << B[i] << std::endl;
            std::cout << i+1 << " " << A[i+1] << " " << B[i+1] << std::endl;
            return false;
        }
    }
    return true;
}

int id(int c, int x, int y, int C, int X, int Y);

int f_id(int a, int c, int x, int y, int A, int C, int X, int Y);

template<typename T>
T ReLU(T v) { return ((v) > 0.0f ? (v) : 0.0f); }

template<typename T>
bool test_conv2D(int Ox, int Oy, 
                 int Fx, int Fy,
                 int C1, int C2,
                 const std::vector<T>& I,
                 const std::vector<T>& F,
                 const std::vector<T>& O,
                 float eps = 1e-3) {
    for (int c2 = 0; c2 < C2; ++c2)
    for (int oy = 0; oy < Oy; ++oy)
    for (int ox = 0; ox < Ox; ++ox) {
        float val = 0.0f;

        for (int c1 = 0; c1 < C1; ++c1)
        for (int fy = 0; fy < Fy; ++fy)
        for (int fx = 0; fx < Fx; ++fx) {
            val += I[id(c1, ox+fx, oy+fy, C1, Ox+Fx-1, Oy+Fy-1)] * F[f_id(c2, c1, fx, fy, C2, C1, Fx, Fy)];
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

template<typename T>
bool test_max_pool(int n, int m, int n1, int m1,
                   const std::vector<T>& A,
                   const std::vector<T>& C,
                   float eps = 1e-7) {

    bool isPassed = true;
    for (size_t i = 0; i < n1; ++i) {
        for(size_t j = 0; j < m1; ++j) {
            float a1 = -1e9,a2 = -1e9,a3 = -1e9,a4 = -1e9;
            if(i * 2 < n && j * 2 < m) {
                a1 = A[i * 2 * m + j * 2];
            }
            if(i * 2 < n && j * 2 + 1 < m) {
                a2 = A[i * 2 * m + j * 2 + 1];
            }
            if((i * 2 + 1) < n && j * 2 < m) {
                a3 = A[(i * 2 + 1) * m + j * 2];
            }
            if((i * 2 + 1) < n && (j * 2 + 1) < m) {
                a4 = A[(i * 2 + 1) * m + (j * 2 + 1)];
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
//        std::cout << std::endl;
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
std::optional<std::vector<T>> matrix_expand(const std::vector<T> & arr,
                              int n, int m,
                              int n1, int m1) {

    if(n > n1 || m > m1) {
        return std::nullopt;
    }

    std::vector<T> fin(n1 * m1, 0);
    size_t iter = 0;
    for(int i = 0; i < n1; ++i) {
        for(int j = 0; j < m1; ++j) {
            if(i < n && j < m) {
                fin[iter] = arr[i * m + j];
            }
            iter++;
        }
    }

    return fin;
}