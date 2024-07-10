#include <cstdio>
#include <iostream>
using namespace std;

__global__ void reduce_sum_kernel(float *d_in, int n, int dim, int tot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tot) {
        d_in[idx] += d_in[idx+tot];  
    }
}

void reduce_sum(float *h_in, float *h_out, size_t n, size_t dim) {
    float* d_in;
    cudaMalloc(&d_in, n*dim*sizeof(float));
    cudaMemcpy(d_in, h_in, n*dim*sizeof(float), cudaMemcpyHostToDevice);

    for (int i = n; i >= 2; i >>= 1) {
        int tot = i*dim/2;
        reduce_sum_kernel<<<(tot+1023)/1024,1024>>>(d_in, i, dim, tot);
    }
    
    cudaMemcpy(h_out, d_in, dim*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in); 
}

const int N = 10000005;
float a[N], b[N];

int main() {
    int n, dim; scanf("%d %d", &n, &dim);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++) {
            a[i*dim+j] = i*dim+j;
            // scanf("%f", &a[i*dim+j]);
        }
    }
    int lim = 1;
    while (lim < n) lim <<= 1;
    
    reduce_sum(a, b, lim, dim);
    for (int i = 0; i < dim; i++) printf("%f ", b[i]);
    return 0;
}