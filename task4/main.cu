#include <cstdio>
#include <iostream>
using namespace std;

#define CUDA_CHECK_ERROR() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(-1); \
    } \
}

const int N = 10000005;
int A[N], B[N];

__global__ void get01_kernel(int *data, int *d01, int n) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < n) {
        d01[id] = (!!data[id]);
    }
}

__global__ void debubble_kernel(int *data, int *ans, int *sum, int n) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < n && data[id] != 0) {
        ans[sum[id]-1] = data[id];
    }
}

const int threadsPerBlock = 512;
const int elementsPerBlock = threadsPerBlock*2;

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

void output(int *a, int n) {
    static int debug[N];
    cudaMemcpy(debug, a, n*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) cout << debug[i] << " "; 
    puts("");
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


int debubble(int *a, int *b, int n) {
    int *data, *d01, *sum, *ans;
    cudaMalloc(&data, n*sizeof(int));
    cudaMalloc(&d01, n*sizeof(int));
    cudaMalloc(&sum, n*sizeof(int));
    cudaMalloc(&ans, n*sizeof(int));
    cudaMemcpy(data, a, n*sizeof(int), cudaMemcpyHostToDevice);

    get01_kernel<<<(n+31)/32, 32>>>(data, d01, n);
    cudaDeviceSynchronize();
    recursive_scan(d01, sum, n);
    cudaDeviceSynchronize();
    debubble_kernel<<<(n+31)/32, 32>>>(data, ans, sum, n);
    cudaDeviceSynchronize();

    cudaMemcpy(b, ans, n*sizeof(int), cudaMemcpyDeviceToHost);
    int ret; 
    cudaMemcpy(&ret, sum+n-1, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(data); cudaFree(d01);
    cudaFree(sum); cudaFree(ans);

    return ret;
}

int main() {
    int n; n = 10000000;
    // scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        // scanf("%d", &A[i]);
        if (i&1) A[i] = i/2+1;
        else A[i] = 0;
    }
    int cnt = 0;
    printf("%d\n", cnt=debubble(A, B, n));
    cout << 1.0*clock()/CLOCKS_PER_SEC << "!\n";
    // for (int i = 0; i < cnt; i++) printf("%d ", B[i]);
    return 0;
}