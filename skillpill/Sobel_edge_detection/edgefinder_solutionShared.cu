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
#include "math.h"
using namespace cimg_library;

struct SPos
{
  int x;
  int y;
  int absOffset;

  __device__ SPos(int xx = 0, int yy = 0, int aO = 0) : x(xx), y(yy), absOffset(aO)
  {
  }
};
__device__ SPos calcOffset(int szX, int szY)
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
    return SPos(-1,-1,-1);
  return SPos(tPosX, tPosY, tPosY * szX + tPosX);
}

__global__ void toGrayscale(float * dOut,
                      const float * dInRed,
                      const float * dInGreen,
                      const float * dInBlue, int szX, int szY)
{
  //Red * 0.3 + Green * 0.59 + Blue * 0.11
  int indx = calcOffset(szX, szY).absOffset;
  if (indx == -1)
    return;
  dOut[indx] = dInRed[indx] * 0.3 + dInGreen[indx] * 0.59 + dInBlue[indx] * 0.11;
  return;
}
__global__ void showBlocksKernel(const float * dInData, float * dOutData, int szX, int szY)
{
  int indx = calcOffset(szX, szY).absOffset;
  if (indx == -1)
    return;
  dOutData[indx] = dInData[indx];
  if (threadIdx.x == 0 || threadIdx.y == 0)
    dOutData[indx] = 0;
  return;
}

__device__ float convolve(const float kernel[][3],
                    int cX, int cY, const float * dInData, int szX, int szY)
{
  float gradient = 0.0f;
  for (int i = -1; i <= 1; i++)
  {
    int curX = cX + i;
    int relX = (curX < 0 ? 0 : (curX >= szX ? szX - 1 : curX));
      for (int j = -1; j <= 1; j++)
      {
        int curY = cY + j;
        int relY = (curY < 0 ? 0 : (curY >= szY ? szY - 1 : curY));
        gradient += kernel[i+1][j+1] * dInData[relX + relY * szX];
      }
  }
  return gradient;
}
__global__ void sobelEdgeDetection(const float * dInData, float * dOutData, int szX, int szY)
{
  SPos pos = calcOffset(szX, szY);
  if (pos.absOffset == -1)
    return;
  __shared__ float sobelKernelX[3][3];/* = {{1, 0, -1},
                            {2, 0, -2},
                            {1, 0, -1}};*/
  __shared__ float sobelKernelY[3][3];/* = {{ 1,  2,  1},
                            { 0,  0,  0},
                            {-1, -2, -1}};*/
  // Shared memory is only seen by the threads in one block!
  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    // If it is the first thread in the block, initialize the filter in shared memory
    sobelKernelX[0][0]=1;sobelKernelX[0][1]=0;sobelKernelX[0][2]=-1;
    sobelKernelX[1][0]=2;sobelKernelX[1][1]=0;sobelKernelX[1][2]=-2;
    sobelKernelX[2][0]=1;sobelKernelX[2][1]=0;sobelKernelX[2][2]=-1;
    sobelKernelY[0][0]=1;sobelKernelY[0][1]=2;sobelKernelY[0][2]=1;
    sobelKernelY[1][0]=0;sobelKernelY[1][1]=0;sobelKernelY[1][2]=0;
    sobelKernelY[2][0]=-1;sobelKernelY[2][1]=-2;sobelKernelY[2][2]=-1;
  }
  __syncthreads();
  float gradientX = convolve(sobelKernelX, pos.x, pos.y, dInData, szX, szY);
  float gradientY = convolve(sobelKernelY, pos.x, pos.y, dInData, szX, szY);
  float totGradient = sqrtf(gradientX * gradientX + gradientY * gradientY);
  dOutData[pos.absOffset] = (float) totGradient;
}

int main(int argc, char *argv[])
{
  // Get file name from the command line
  std::string fileName("images/thebrain.png");
  if (argc > 1)
    fileName = std::string(argv[1]);
  std::cout << fileName;
  // Load the image
  CImg<float> image(fileName.c_str());
  // Get image dimensions
  int szX = image.width();
  int szY = image.height();
  std::cout << "Size of the image: " << szX << " x " << szY << std::endl;
  // Extract the channels
  const float * redChannel = &image(0, 0, 0, 0);
  const float * greenChannel = &image(0, 0, 0, 1);
  const float * blueChannel = &image(0, 0, 0, 2);
  // Create the output image as a grayscale image of the same size as input
  CImg<float> outImage(szX, szY, 1, 1, 0);
  // Extract the data from the output image
  float * outGray = &outImage(0, 0, 0, 0);
  // Get cuda properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  // Maximum threads per block on this device
  int maxThreads = prop.maxThreadsPerBlock;
  std::cout << "Maximum threads per block: " << maxThreads;
  // Allocate memory on the GPU
  int dataSz = szX * szY * sizeof(float);
  float * d_redCh = NULL;
  float * d_greenCh = NULL;
  float * d_blueCh = NULL;
  checkCudaErrors(cudaMalloc(&d_redCh, dataSz));
  checkCudaErrors(cudaMalloc(&d_greenCh, dataSz));
  checkCudaErrors(cudaMalloc(&d_blueCh, dataSz));
  float * d_gray = NULL;
  checkCudaErrors(cudaMalloc(&d_gray, dataSz));
  float * d_outGray = NULL;
  checkCudaErrors(cudaMalloc(&d_outGray, dataSz));
  // Copy image data to GPU
  checkCudaErrors(cudaMemcpy(d_redCh, redChannel, dataSz, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_greenCh, greenChannel, dataSz, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_blueCh, blueChannel, dataSz, cudaMemcpyHostToDevice));
  // Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize((int) sqrt(maxThreads), (int) sqrt(maxThreads), 1);
  // Compute correct grid size (i.e., number of blocks per kernel launch)
  // from the image size and and block size.
  const dim3 gridSize(int(szX/blockSize.x)+1, int(szY/blockSize.y)+1, 1);
  // Convert image to grayscale
  toGrayscale<<<gridSize, blockSize>>>(d_gray, d_redCh, d_greenCh, d_blueCh, szX, szY);
  // Run show blocks kernel
  showBlocksKernel<<<gridSize, blockSize>>>(d_gray, d_outGray, szX, szY);
  // Copy results back to CPU
  checkCudaErrors(cudaMemcpy(outGray, d_outGray, dataSz, cudaMemcpyDeviceToHost));
  // Save resulting image
  outImage.save_png("images/testblocks.png");
  // Detect edges using Sobel operator
  sobelEdgeDetection<<<gridSize, blockSize, 9*2*sizeof(float)>>>(d_gray, d_outGray, szX, szY);
  // Copy results back to CPU
  checkCudaErrors(cudaMemcpy(outGray, d_outGray, dataSz, cudaMemcpyDeviceToHost));
  float minVal = 0.0;
  float maxVal = outImage.max_min(minVal);
  // Normalize
  outImage.operator*=(255/maxVal);
  // Save the result
  outImage.save_png("images/testedge.png");
  // Clean up GPU memory, we don't need the color channels anymore
  checkCudaErrors(cudaFree(d_redCh));
  checkCudaErrors(cudaFree(d_greenCh));
  checkCudaErrors(cudaFree(d_blueCh));
  checkCudaErrors(cudaFree(d_outGray));
  checkCudaErrors(cudaFree(d_gray));
}
