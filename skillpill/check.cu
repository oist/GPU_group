/*------------check.cu------------------------------------------------------//
*
* Purpose: This is a simple cuda file for checking your gpu works
*
*          It prints 0 -> 63
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <math.h>

__global__ void findID(double *a, int n){

    // First we need to find our global threadID
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    // Make sure we are not out of range
    if (id < n){
        a[id] = id;
    }
}

int main(){

    // size of vectors
    int n = 64;

    // Host vectors
    double *h_a;

    // Device vectors
    double *d_a;

    // allocating space on host and device
    h_a = (double*)malloc(sizeof(double)*n);

    // Allocating space on GPU
    cudaMalloc(&d_a, sizeof(double)*n);

    // Creating blocks and grid ints
    int threads, grid;

    threads = 64;
    grid = (int)ceil((float)n/threads);

    findID<<<grid, threads>>>(d_a, n);

    // Now to copy c back
    cudaMemcpy(h_a, d_a, sizeof(double)*n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i){
        std::cout << h_a[i] << '\n';
    }

    // Release memory
    cudaFree(d_a);

    free(h_a);
}
