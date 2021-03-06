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
  /*
  * TODO: implement appropriate indexing
  */
  return SPos(0, 0, 0);
}

__global__ void toGrayscale(float * dOut,
                      const float * dInRed,
                      const float * dInGreen,
                      const float * dInBlue, int szX, int szY)
{
  //Red * 0.3 + Green * 0.59 + Blue * 0.11
  /*
  * TODO: implement convertion to grayscale
  */
  return;
}
__global__ void showBlocksKernel(const float * dInData, float * dOutData, int szX, int szY)
{
  // This is just to show how do the blocks look like
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
  /*
  * TODO: implement convolution
  */
  return 0.0;
}
__global__ void sobelEdgeDetection(const float * dInData, float * dOutData, int szX, int szY)
{
  /*
  * TODO: implement Sobel filtering
  */
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
  std::cout << "Maximum threads per block: " << maxThreads << std::endl;
  // Allocate memory on the GPU
  int dataSz = szX * szY * sizeof(float);
  float * d_redCh = NULL;
  checkCudaErrors(cudaMalloc(&d_redCh, dataSz));
  /*
  * TODO: Allocate more channels here
  */
  // Copy image data to GPU
  /*
  * TODO: Copy input data to the GPU: CPU->GPU copy
  */
  // Set reasonable block size (i.e., number of threads per block)
  /*
  * TODO: Your code here>> const dim3 blockSize(..., ..., 1);
  */
  const dim3 blockSize(1, 1, 1);
  // Compute correct grid size (i.e., number of blocks per kernel launch)
  // from the image size and and block size.
  /*
  * TODO: Your code here
  */
  const dim3 gridSize(1, 1, 1);
  // Convert image to grayscale
  /*
  * TODO: Call the RBG->grayscale kernel
  */
  // Run show blocks kernel
  /*
  * TODO: Run the kernel
  */
  // Copy results back to CPU
  /*
  * TODO: Your code here - > GPU->CPU copy
  */
  // Save resulting image
  outImage.save_png("images/testblocks.png");
  // Detect edges using Sobel operator
  /*
  * TODO: Run the filter kernel
  */
  // Copy results back to CPU
  /*
  * TODO: Copy GPU->CPU
  */
  // Normalize
  float minVal = 0.0;
  float maxVal = outImage.max_min(minVal);
  outImage.operator*=(255/maxVal);
  // Save the result
  outImage.save_png("images/testedge.png");
  // Clean up GPU memory, we don't need the color channels anymore
  /*
  * TODO: Free the memory of all the channels
  */
  checkCudaErrors(cudaFree(d_redCh));
}
