/**
* This file contains the main function of the edge detection
* program.
* Author: Irina Reshodko
* For GPU Skill Pill @ OIST, Nov. 2017
*/
#include <iostream>
#include "CIMG/CImg.h"

using namespace cimg_library;

int main(int argc, char *argv[])
{
  // Load the image
  CImg<unsigned char> image("images/thebrain.png");
  // Get image dimensions
  int szX = image.width();
  int szY = image.height();
  // Extract the channels
  const unsigned char * redChannel = &image(0, 0, 0, 0);
  const unsigned char * greenChannel = &image(0, 0, 0, 1);
  const unsigned char * blueChannel = &image(0, 0, 0, 2);
  // Create the output image as a 3-channel image of the same size as input
  CImg<unsigned char> outImage(szX, szY, 1, 3);
  // Extract the channels form the output image
  unsigned char * outRedCh = &outImage(0, 0, 0, 0);
  unsigned char * outGreenCh = &outImage(0, 0, 0, 1);
  unsigned char * outBlueCh = &outImage(0, 0, 0, 2);
  outRedCh[20] = 200;
  outGreenCh[30] = 200;
  outBlueCh[10] = 200;
  outImage.save_png("images/test.png");
}
