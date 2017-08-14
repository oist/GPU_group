/*------------vec_add.cu------------------------------------------------------//
*
* Purpose: This is a simple cuda file for vector addition
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <math.h>

__global__ void vecAdd(double *a, double *b, double *c, int n){

    // First we need to find our global threadID
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    // Make sure we are not out of range
    if (id < n){
        c[id] = a[id] + b[id];
    }
}

int main(){

    // size of vectors
    int n = 1000;

    // Host vectors
    double *h_a, *h_b, *h_c;

    // Device vectors
    double *d_a, *d_b, *d_c;

    // allocating space on host and device
    h_a = (double*)malloc(sizeof(double)*n);
    h_b = (double*)malloc(sizeof(double)*n);
    h_c = (double*)malloc(sizeof(double)*n);

    // Allocating space on GPU
    cudaMalloc(&d_a, sizeof(double)*n);
    cudaMalloc(&d_b, sizeof(double)*n);
    cudaMalloc(&d_c, sizeof(double)*n);

    //initializing host vectors
    for (int i = 0; i < n; ++i){
        h_a[i] = 1;
        h_b[i] = 1;
    }

    // copying these components to the GPU
    cudaMemcpy(d_a, h_a, sizeof(double)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(double)*n, cudaMemcpyHostToDevice);

    // Creating blocks and grid ints
    int threads, grid;

    threads = 64;
    grid = (int)ceil((float)n/threads);

    vecAdd<<<grid, threads>>>(d_a, d_b, d_c, n);

    // Now to copy c back
    cudaMemcpy(h_c, d_c, sizeof(double)*n, cudaMemcpyDeviceToHost);

    double sum = 0;
    for (int i = 0; i < n; ++i){
        sum += h_c[i];
    }

    std::cout << "Sum is: " << sum << '\n';

    // Release memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);
}
