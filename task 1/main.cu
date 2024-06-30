#include <cstdio>
#include <iostream>
using namespace std;

__global__ void reduce_sum_kernel(const float* input_vecs, size_t n, size_t dim, float* output_vec) {
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < dim) {
        float sum = 0;
        for (int i = 0; i < n; i++) 
            sum += input_vecs[i*dim+k];
        output_vec[k] = sum;
    }
}


void reduce_sum(const float* input_vecs, size_t n, size_t dim, float* output_vec) {
    size_t threads_per_block = 32;
    size_t num_of_blocks = (dim-1)/threads_per_block+1;
    reduce_sum_kernel<<<num_of_blocks,threads_per_block>>>(input_vecs, n, dim, output_vec);
}

const int N = 1005;
float a[N], b[N];

int main() {
    int n, dim; scanf("%d %d", &n, &dim);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++)
            scanf("%f", &a[i*dim+j]);
    }

    float* d_input_vecs;
    float* d_output_vec;
    cudaMalloc((void**)&d_input_vecs, n*dim*sizeof(float));
    cudaMalloc((void**)&d_output_vec, dim*sizeof(float));
    cudaMemcpy(d_input_vecs, a, n*dim*sizeof(float), cudaMemcpyHostToDevice);
    reduce_sum(d_input_vecs, n, dim, d_output_vec);
    cudaMemcpy(b, d_output_vec, dim*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < dim; i++) printf("%f ", b[i]);

    cudaFree(d_input_vecs); 
    cudaFree(d_output_vec);
    return 0;
}