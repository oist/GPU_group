/**
* This file contains the code of the edge detection example
* program.
* Author: Irina Reshodko
* For GPU Skill Pill @ OIST, Nov. 2017
*/
#include <iostream>
#include <string>
#include "CIMG/CImg.h"
#include <cuda_runtime.h>
#include "utils.h"
using namespace cimg_library;

__device__ dim3 calcOffset(int szX, int szY)
{
  // 2D grid with 2D blocks
  // int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	// int indx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
  // if (indx >= szX * szY)
  //   return -1;
  // return indx;
  // Calculate appropriate offsets in the array
  int tPosX = blockIdx.x * blockDim.x + threadIdx.x;
  int tPosY = blockIdx.y * blockDim.y + threadIdx.y;
  if (tPosX >= szX || tPosY >= szY)
    return dim3(-1,-1,-1);
  return dim3(tPosX, tPosY, tPosY * szX + tPosX);
}
__global__ void showBlocksKernel(const unsigned char * dInData, unsigned char * dOutData, int szX, int szY)
{
  int indx = calcOffset(szX, szY).z;
  dOutData[indx] = dInData[indx];
  if (threadIdx.x == 0 || threadIdx.y == 0)
    dOutData[indx] = 0;
  return;
}

__device__ convolve(const float * kernel, int kernelSzX, int kernelSzY,
                    int cX, int cY, const unsigned char * dInData, int szX, int szY)
{
  float gradient = 0.0f;
  for (int i = -kernelSzX/2; i <= kernelSzX/2; i++)
  {
    int curX = cX + i;
    int relX = (curX < 0 ? 0 : (curX >= szX ? szX - 1 : curX));
      for (int j = -kernelSzY/2; j <= kernelSzY/2; j++)
      {
        int curY = cY + j;
        int relY = (curY < 0 ? 0 : (curY >= szY ? szY - 1 : curY));
        gradient += kernel[i][j] * dInData[curX + curY * szX];
      }
  }
  return gradient;
}
__global__ void sobelEdgeDetection(const unsigned char * dInData, unsigned char * dOutData, int szX, int szY)
{
  int pos = calcOffset(szX, szY);
  float sobelKernelX[3][3] = {{1, 0, -1},
                            {2, 0, -2},
                            {1, 0, -1}};
  float sobelKernelY[3][3] = {{ 1,  2,  1},
                            { 0,  0,  0},
                            {-1, -2, -1}};
  int fromX = 0;
  int fromY = 0;
  int toX = 0;
  int toY = 0;
  float gradientX = convolve(sobelKernelX, 3, 3, pos.x, pos.y, dInData, szX, szY);
  float gradientY = convolve(sobelKernelY, 3, 3, pos.x, pos.y, dInData, szX, szY);
  float totGradient = sqrt(gradientX * gradientX + gradientY * gradientY)/sqrt(2);
  dOutData[pos.z] = (unsigned char) totGradient;
}

int main(int argc, char *argv[])
{
  // Get file name from the command line
  std::string fileName("images/thebrain.png");
  if (argc > 1)
    fileName = std::string(argv[2]);
  std::cout << fileName;
  // Load the image
  CImg<unsigned char> image(fileName.c_str());
  // Get image dimensions
  int szX = image.width();
  int szY = image.height();
  // Extract the channels
  const unsigned char * redChannel = &image(0, 0, 0, 0);
  const unsigned char * greenChannel = &image(0, 0, 0, 1);
  const unsigned char * blueChannel = &image(0, 0, 0, 2);
  // Create the output image as a 3-channel image of the same size as input
  CImg<unsigned char> outImage(image, false);
  // Extract the channels form the output image
  unsigned char * outRedCh = &outImage(0, 0, 0, 0);
  unsigned char * outGreenCh = &outImage(0, 0, 0, 1);
  unsigned char * outBlueCh = &outImage(0, 0, 0, 2);
  // Test if it works
  outRedCh[20] = 200;
  outGreenCh[30] = 200;
  outBlueCh[10] = 200;
  outImage.save_png("images/test.png");
  // Get cuda properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  // Maximum threads per block on this device
  int maxThreads = prop.maxThreadsPerBlock;
  std::cout << "Maximum threads per block: " << maxThreads;
  // Allocate memory on the GPU
  int dataSz = szX * szY * sizeof(unsigned char);
  unsigned char * d_redCh = NULL;
  unsigned char * d_greenCh = NULL;
  unsigned char * d_blueCh = NULL;
  checkCudaErrors(cudaMalloc(&d_redCh, dataSz));
  checkCudaErrors(cudaMalloc(&d_greenCh, dataSz));
  checkCudaErrors(cudaMalloc(&d_blueCh, dataSz));
  unsigned char * d_outRedCh =NULL;
  unsigned char * d_outGreenCh = NULL;
  unsigned char * d_outBlueCh = NULL;
  checkCudaErrors(cudaMalloc(&d_outRedCh, dataSz));
  checkCudaErrors(cudaMalloc(&d_outGreenCh, dataSz));
  checkCudaErrors(cudaMalloc(&d_outBlueCh, dataSz));
  // Copy image data to GPU
  checkCudaErrors(cudaMemcpy(d_redCh, redChannel, dataSz, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_greenCh, greenChannel, dataSz, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_blueCh, blueChannel, dataSz, cudaMemcpyHostToDevice));
  // Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize((int) sqrt(maxThreads), (int) sqrt(maxThreads), 1);
  // Compute correct grid size (i.e., number of blocks per kernel launch)
  // from the image size and and block size.
  const dim3 gridSize(int(szX/blockSize.x)+1, int(szY/blockSize.y)+1, 1);
  // Run show blocks kernel
  showBlocksKernel<<<gridSize, blockSize>>>(d_redCh, d_outRedCh, szX, szY);
  showBlocksKernel<<<gridSize, blockSize>>>(d_greenCh, d_outGreenCh, szX, szY);
  showBlocksKernel<<<gridSize, blockSize>>>(d_blueCh, d_outBlueCh, szX, szY);
  // Copy results back to CPU
  checkCudaErrors(cudaMemcpy(outRedCh, d_outRedCh, dataSz, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(outGreenCh, d_outGreenCh, dataSz, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(outBlueCh, d_outBlueCh, dataSz, cudaMemcpyDeviceToHost));
  // Save resulting image
  outImage.save_png("images/testblocks.png");

  // Clean up GPU memory
  checkCudaErrors(cudaFree(d_redCh));
  checkCudaErrors(cudaFree(d_greenCh));
  checkCudaErrors(cudaFree(d_blueCh));
  checkCudaErrors(cudaFree(d_outRedCh));
  checkCudaErrors(cudaFree(d_outGreenCh));
  checkCudaErrors(cudaFree(d_outBlueCh));
}
