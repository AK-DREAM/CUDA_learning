#include <cstdio>
#include <iostream>
using namespace std;

__global__ void matmulKernel(const float* P, const float* Q, size_t m, size_t n, size_t k, float* R) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < m && y < n) {
        float sum = 0;
        for (int i = 0; i < k; i++) {
            sum += P[x*k+i]*Q[i*n+y];
        }
        R[x*n+y] = sum;
    }
}

void matmul(const float* P, const float* Q, size_t m, size_t n, size_t k, float* R) {
    float* A2;
    float* B2;
    float* C2;
    cudaMalloc((void**)&A2, m*k*sizeof(float));
    cudaMalloc((void**)&B2, k*n*sizeof(float));
    cudaMalloc((void**)&C2, m*n*sizeof(float));
    cudaMemcpy(A2, P, m*k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B2, Q, k*n*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m-1)/threadsPerBlock.x+1, (n-1)/threadsPerBlock.y+1); 
    matmulKernel<<<numBlocks, threadsPerBlock>>>(P, Q, m, n, k, R);

    cudaMemcpy(R, C2, m*n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A2); 
    cudaFree(B2);
    cudaFree(C2);
}

const int N = 2005;
float A[N*N], B[N*N], C[N*N], tmp[N*N];
float dA[N*N], dB[N*N], dC[N*N];

int main() {
    int m, n, k; scanf("%d %d %d", &m, &n, &k);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            scanf("%f", &A[i*k+j]);
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%f", &B[i*n+j]);
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%f", &dC[j*k+i]);
        }
    }
    matmul(A, B, m, n, k, C);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f%c", C[i*n+j], " \n"[j==n-1]);
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            tmp[j*k+i] = B[i*n+j];
        }
    }
    matmul(dC, tmp, m, k, n, dA);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            tmp[j*m+i] = A[i*k+j];
        }
    }
    matmul(tmp, dC, k, n, m, dB);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            printf("%f%c", dA[i*k+j], " \n"[j==k-1]);
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f%c", dB[i*n+j], " \n"[j==n-1]);
        }
    }
    return 0;
}