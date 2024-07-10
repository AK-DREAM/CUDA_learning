#include <iostream>
#include <cstdio>
#define N 100000005
using namespace std;

#define CUDA_CHECK_ERROR() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(-1); \
    } \
}

void output(int *a, int n) {
    static int debug[N];
    cudaMemcpy(debug, a, n*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) cout << debug[i] << " "; 
    puts("");
}

const int threadsPerBlock = 512;
const int elementsPerBlock = threadsPerBlock*2;

int a[N];

__global__ void scan_kernel(int *data, int *sum, int *bsum, int n) {
    __shared__ int tmp[elementsPerBlock];
    int tid = threadIdx.x, bid = blockIdx.x;
    int ofst = bid*elementsPerBlock;

    tmp[tid*2] = ofst+tid*2<n?data[ofst+tid*2]:0;
    tmp[tid*2+1] = ofst+tid*2+1<n?data[ofst+tid*2+1]:0;
    __syncthreads();

    int step = 1;
    for (int d = elementsPerBlock>>1; d; d >>= 1) {
        if (tid < d) {
            tmp[step*(2*tid+2)-1] += tmp[step*(2*tid+1)-1];
        }
        step <<= 1;
        __syncthreads();
    }
    if (!tid) bsum[bid] = tmp[elementsPerBlock-1];
    for (int d = 1; d < elementsPerBlock; d <<= 1) {
        step >>= 1;
        if (tid < d-1) {
            tmp[step*(2*tid+3)-1] += tmp[step*(2*tid+2)-1];
        }
        __syncthreads();
    }

    if (ofst+tid*2 < n) sum[ofst+tid*2] = tmp[tid*2];
    if (ofst+tid*2+1 < n) sum[ofst+tid*2+1] = tmp[tid*2+1];
}

__global__ void add_kernel(int *sum, int *bsum2, int n) {
    int tid = threadIdx.x, bid = blockIdx.x;
    int ofst = bid*elementsPerBlock;
    if (bid > 0 && ofst+tid*2 < n) sum[ofst+tid*2] += bsum2[bid-1];
    if (bid > 0 && ofst+tid*2+1 < n) sum[ofst+tid*2+1] += bsum2[bid-1];
}

void recursive_scan(int *data, int *sum, int n) {
    int blockNum = (n+elementsPerBlock-1)/elementsPerBlock;
    int *bsum, *bsum2;
    cudaMalloc(&bsum, blockNum*sizeof(int));
    cudaMalloc(&bsum2, blockNum*sizeof(int));

    scan_kernel<<<blockNum, threadsPerBlock>>>(data, sum, bsum, n);

    if (blockNum > 1) {
        recursive_scan(bsum, bsum2, blockNum);
        add_kernel<<<blockNum, threadsPerBlock>>>(sum, bsum2, n);
    }
}

__global__ void work1_kernel(int *d_in, int *d_tmp, int n, int i) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; 
    if (idx < n) {
        d_tmp[idx] = !(d_in[idx]>>i&1);
    }
}

__global__ void work2_kernel(int *d_in, int *d_tmp, int *d_out, int n, int i, int tot) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; 
    if (idx < n) {
        int nw = idx>0?d_tmp[idx-1]:0;
        int pos = (d_in[idx]>>i&1)?idx+tot-nw:nw;
        d_out[pos] = d_in[idx];
    }
    __syncthreads();
    if (idx < n) {
        d_in[idx] = d_out[idx];
    }
}

void sort(int *h_num, int n) {
    int *d_in, *d_out, *d_tmp;
    cudaMalloc(&d_in, n*sizeof(int));
    cudaMalloc(&d_tmp, n*sizeof(int));
    cudaMalloc(&d_out, n*sizeof(int));
    cudaMemcpy(d_in, h_num, n*sizeof(int), cudaMemcpyHostToDevice);

    double t = 1.0*clock()/CLOCKS_PER_SEC;
    for (int i = 0; i < 31; i++) {
        int blockNum = (n+threadsPerBlock-1)/threadsPerBlock;

        work1_kernel<<<blockNum, threadsPerBlock>>>(d_in, d_tmp, n, i);
        cudaDeviceSynchronize();

        recursive_scan(d_tmp, d_tmp, n);
        cudaDeviceSynchronize();

        int tot; cudaMemcpy(&tot, &d_tmp[n-1], sizeof(int), cudaMemcpyDeviceToHost);
        work2_kernel<<<blockNum, threadsPerBlock>>>(d_in, d_tmp, d_out, n, i, tot);
        cudaDeviceSynchronize();
    }
    printf("%lf\n", 1.0*clock()/CLOCKS_PER_SEC-t);

    cudaMemcpy(h_num, d_out, n*sizeof(int), cudaMemcpyDeviceToHost);
}

int main() {
    int n = 10000000;
    for (int i = 0; i < n; i++) {
        a[i] = rand();
    }
    double t = 1.0*clock()/CLOCKS_PER_SEC;
    sort(a, n);
    printf("%lf\n", 1.0*clock()/CLOCKS_PER_SEC-t);
    //for (int i = 0; i < n; i++) printf("%d ", a[i]);
    return 0;
}