#include<cuda_runtime.h>
#include<iostream>
#include<vector>

__global__
void vecAdditionKernel(float* A, float* B, float* C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        printf("Block: %d, Thread: %d, Dim: %d, Global Index: %d\n", blockIdx.x, threadIdx.x, blockDim.x, i);
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n){
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1)/threadsPerBlock;

    vecAdditionKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {

    int n = 256;

    float *A_h, *B_h, *C_h;

    A_h = new float[n];
    B_h = new float[n];
    C_h = new float[n];

    for (int i = 0; i < n; i++) {
        A_h[i] = static_cast<float>(i);
        B_h[i] = static_cast<float>(2 * i);
    }
    
    std::cout<<"Executing addition";
    vecAdd(A_h, B_h, C_h, n);

    for (int i = 0; i < n; i++) {
        std::cout << "C_h[" << i << "] = " << C_h[i] << " checked" << std::endl;
    }

    delete[] A_h;
    delete[] B_h;
    delete[] C_h;

    return 0;
}

