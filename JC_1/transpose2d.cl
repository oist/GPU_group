/*-------------transpose2d.cl-------------------------------------------------//
*
* Purpose: to reimplement the 2d transpose notated here:
*              http://developer.download.nvidia.com/compute/DevZone/C/html_x64/6
_Advanced/transpose/doc/MatrixTranspose.pdf
*
*-----------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void transpose2d(__global double *in, __global double *out,
                          const unsigned int width, const unsigned int height){
}
