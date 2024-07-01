#include <cstdio>
#include <iostream>
using namespace std;

const int N = 1000005;
int a[N];

__global__ void sort_kernel(int *d_vec, int n, int tp) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    i = 2*i+tp;
    if (i+1 < n) {
        if (d_vec[i] > d_vec[i+1]) {
            int tmp = d_vec[i];
            d_vec[i] = d_vec[i+1];
            d_vec[i+1] = tmp;
        }
    }
}

void sort(int *a, int n) {
    int* d_vec;
    cudaMalloc((void**)&d_vec, n*sizeof(int));
    cudaMemcpy(d_vec, a, n*sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < n; i++) {
        sort_kernel<<<(n+127)/128,128>>>(d_vec, n, i&1);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(a, d_vec, n*sizeof(int), cudaMemcpyDeviceToHost);
}

int main() {
    int n; scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &a[i]);
    }
    sort(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    return 0;
}