/*-------------copy2d.cl------------------------------------------------------//
*
* Purpose: to reimplement the 2d copy notated here:
*              http://developer.download.nvidia.com/compute/DevZone/C/html_x64/6_Advanced/transpose/doc/MatrixTranspose.pdf
*
*-----------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void copy2d(__global double *in, __global double *out,
                     const unsigned int width, const unsigned int height,
                     const unsigned int tile, const unsigned int block_rows){
    // Global Thread ID
    int xid = get_global_id(0);
    int yid = get_global_id(1);

    int index = xid + width*yid;
    for (int i = 0; i < tile;  i+=block_rows){
        out[index + i*width] = in[index + i*width];
    }


}

