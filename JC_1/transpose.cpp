/*-------------transpose.cpp--------------------------------------------------//
*
* Purpose: to reimplement the 2d transpose notated here:
*              http://developer.download.nvidia.com/compute/DevZone/C/html_x64/6_Advanced/transpose/doc/MatrixTranspose.pdf
*          and the 3d transpose implemented here:
*              https://link.springer.com/article/10.1007/s10766-015-0366-5
*
*-----------------------------------------------------------------------------*/

# define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>

#define TILE 32
#define BLOCK 8

// Function to test the copy kernel
double *test_copy(double *array, int width, int height);

// Function to test the transpose kernel
double *test_transpose(double *array, int width, int height);

// Function to print an array
void print_array(double *array, int width, int height);

// Function to open a file and return a const char *
const char *open_file(std::string filename);

/*
const char *copy_kernel = "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n" \
"__kernel void copy2d(__global double *in, __global double *out, \n" \
                     "const unsigned int width, const unsigned int height \n" \
                     "const unsigned int tile, const unsigned int block_rows){ \n" \
    "// Global Thread ID \n" \
    "int xid = get_global_id(0); \n" \
    "int yid = get_global_id(1); \n" \
" \n" \
    "int index = xid + width*yid; \n" \
    "for (int i = 0; i < tile;  i+=block_rows){ \n" \
        "out[index + i*width] = in[index + i*width]; \n" \
    "} \n" \
" \n" \
"} \n" \
" \n" ;
*/

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

// main function
int main(){

    // Creating 32x32 array to mess around with
    double array[32*32];
    for (int i = 0; i < 32*32; i++){
        if (i % 2 == 0){
            array[i] = 0.0;
        }
        else{
            array[i] = 1.0;
        }
    }

    print_array(array, 32, 32);

    std::cout << "\n\n";

    double *copied_array = test_copy(array, 32, 32);
    print_array(copied_array, 32, 32);
    
}


/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to test the copy kernel
double *test_copy(double *array, int width, int height){

    double *out;
    out = new double[width*height];

    cl::Buffer d_out, d_in;
    cl_int err = CL_SUCCESS;
    try{
        // First, query for platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0){
            std::cout << "Platform Size 0!\n";
            exit(1);
        }

        // Get list of devices on default platform and creat context
        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Create command queue
        cl::CommandQueue queue(context, devices[0], 0, &err);

        // Creating device memory buffer
        d_in = cl::Buffer(context, CL_MEM_READ_ONLY, 
                          width*height*sizeof(double));
        d_out = cl::Buffer(context, CL_MEM_WRITE_ONLY, 
                           width*height*sizeof(double));

        // Bind memory buffer
        queue.enqueueWriteBuffer(d_in, CL_TRUE, 0, width*height*sizeof(double),
                                 array);

        // now to read in the kernel
        const char *copy_kernel = open_file("copy2d.cl");

        std::cout << copy_kernel << '\n';
        cl::Program::Sources source(1, 
            std::make_pair(copy_kernel, strlen(copy_kernel)));

        cl::Program program = cl::Program(context, source);
        err = program.build(devices);
        if (err != CL_SUCCESS){
            std::cout << "you got problems" << '\n';
        }
        cl::Kernel kernel(program, "copy2d", &err);

        // Bind arguments for kernel
        kernel.setArg(0,d_in);
        kernel.setArg(1,d_out);
        kernel.setArg(2,width);
        kernel.setArg(3,height);
        kernel.setArg(4,TILE);
        kernel.setArg(5,BLOCK);

        // Number of work-items
        cl::NDRange localSize(64);

        cl::NDRange globalSize((int)(ceil(width*height/(float)64)*64));

        cl::Event event;
        queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            globalSize,
            localSize,
            NULL,
            &event
        );

        event.wait();

        queue.enqueueReadBuffer(d_out, CL_TRUE, 0, width*height*sizeof(double),
                                out);

        
    }
    catch(cl::Error err){
        std::cerr << "ERROR: " << err.what() 
                  << "("<<err.err()<<")" << std::endl;

    }

    return out;
}

// Function to test the transpose kernel
double *test_transpose(double *array, int width, int height){
}

// Function to print an array
void print_array(double *array, int width, int height){

    int index = 0;
    for (int i = 0; i < height; ++i){
        for (int j = 0; j < width; j++){
            index = j + i*width;
            std::cout << array[index] << ' ';
        }
        std::cout << '\n';
    }
}

// Function to open a file and return a const char *
const char *open_file(std::string filename){
    FILE *f;
    char *source;
    size_t src_size, program_size;

    f = fopen(filename.c_str(), "rb");
    if (!f){
        std::cout << "Failed to load kernel " << filename << "!\n";
        exit(1);
    }

    fseek(f,0,SEEK_END);
    program_size = ftell(f);
    rewind(f);
    source = (char*)malloc(program_size*sizeof(char)+1);
    source[program_size] = '\0';

    fread(source, sizeof(char), program_size, f);
    fclose(f);
    return source;
}
