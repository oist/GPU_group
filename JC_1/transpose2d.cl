/*-------------transpose2d.cl-------------------------------------------------//
*
* Purpose: to reimplement the 2d transpose notated here:
*              http://developer.download.nvidia.com/compute/DevZone/C/html_x64/6_Advanced/transpose/doc/MatrixTranspose.pdf
*
*-----------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void transpose2d(__global int *in, __global int *out, 
                          __local int *tile,
                          const unsigned int width, const unsigned int height,
                          const unsigned int tile_size, 
                          const unsigned int block_rows){
    // STRAIGHT FORWARD EXAMPLE
    int xid = get_global_id(0);
    int yid = get_global_id(1);

    int index_in = xid + width*yid;
    int index_out = yid + height*xid;

    out[index_out] = in[index_in];

/*
    // TILE EXAMPLE
    // Global Thread ID
    int xid = get_group_id(0)*tile_size + get_local_id(0);
    int yid = get_group_id(1)*tile_size + get_local_id(1);


    int index_in = xid + width*yid;
    int index_out = yid + height*xid;
    for (int i = 0; i < tile;  i+=block_rows){
        out[index_out + i] = in[index_in + i*width];
    }
*/
/*
    // COALESCED TRANSPOSE
    int xthread = get_local_id(0);
    int ythread = get_local_id(1);
    int xid = get_group_id(0)*tile_size + xthread;
    int yid = get_group_id(1)*tile_size + ythread;

    int index_in = xid + yid * width;

    xid = get_group_id(1) * tile_size + xthread;
    yid = get_group_id(0) * tile_size + ythread;

    int index_out = xid + yid * height;

    int tile_index = 0;
    for (int i = 0; i < tile_size; i += block_rows){
        tile_index = (ythread+i)*width + xthread;
        tile[tile_index] = in[index_in + width];
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for (int i = 0; i < tile_size; i += block_rows){
        tile_index = ythread + i + xthread*height;
        out[index_out + i*height] = tile[tile_index];
    }
*/
/*
    int blockIdx_x, blockIdx_y;

    if (width == height){
        blockIdx_y = get_group_id(0);
        blockIdx_x = (get_group_id(0)+get_group_id(1))%get_global_size(0);
    }
    else{
        int bid = get_group_id(0) + get_global_size(0)*get_group_id(1);
        blockIdx_y = bid%get_global_size(1);
        blockIdx_x = ((bid/get_global_size(1))+get_group_id(1))
                     %get_global_size(0);
    }
    int xthread = get_local_id(0);
    int ythread = get_local_id(1);
    int xid = blockIdx_x*tile_size + xthread;
    int yid = blockIdx_y*tile_size + ythread;
    int index_in = xid + width*yid;

    xid = blockIdx_y*tile_size + xthread;
    yid = blockIdx_x*tile_size + ythread;
    int index_out = xid + height*yid;

    int tile_index = 0;
    for (int i = 0; i < tile_size; i += block_rows){
        tile_index = (ythread+i)*width + xthread;
        tile[tile_index] = in[index_in + i*width];
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for (int i = 0; i < tile_size; i += block_rows){
        tile_index = ythread + i + xthread*width;
        out[index_out + i*height] = tile[tile_index];
    }
*/
}
