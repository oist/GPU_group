#include "CIMG/CImg.h"
using namespace cimg_library;
int main() {
  CImg<unsigned char> image("images/thebrain.png"), visu(500,400,1,3,0);
  const unsigned char red[] = { 255,0,0 }, green[] = { 0,255,0 }, blue[] = { 0,0,255 };
  image.blur(2.5);
  image.save_png("thebrain_blur.png");
  return 0;
}
