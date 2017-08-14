/*-------------vec_add.cpp----------------------------------------------------//
*
* Purpose: This is a simple vector addition code in OpenCL
*
*   Notes: This is heavily influenced by:
*              https://www.olcf.ornl.gov/tutorials/opencl-vector-addition/
*
*-----------------------------------------------------------------------------*/

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <math.h>

// OpenCL kernel
const char *kernelSource =                                   "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                \n" \
"__kernel void vecAdd( __global double *a,                    \n" \
"                      __global double *b,                    \n" \
"                      __global double *c,                    \n" \
"                      const unsigned int n){                 \n" \
"    // Global Thread ID                                      \n" \
"    int id = get_global_id(0);                               \n" \
"                                                             \n" \
"    // Remain in bounds...                                   \n" \
"    if (id < n){                                             \n" \
"        c[id] = a[id] + b[id];                               \n" \
"    }                                                        \n" \
"}                                                            \n" \
                                                             "\n" ;

// Now for the main function
int main(){

    // length of vectors
    unsigned int n = 1000000;

    // host vectors
    double *h_a, *h_b, *h_c;

    h_a = new double[n];
    h_b = new double[n];
    h_c = new double[n];

    // Setting all elements to 1
    for (int i = 0; i < n; ++i){
        h_a[i] = 1;
        h_b[i] = 1;
    }

    // device vectors
    cl::Buffer d_a, d_b, d_c;

    cl_int err = CL_SUCCESS;
    try{

        // First, Query for platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0){
            std::cout << "Platform size 0\n";
            return -1;
        }

        // Get list of devices on default platform and create context
        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};

        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Create command queue on first device
        cl::CommandQueue queue(context, devices[0], 0, &err);

        // Create device memory buffers
        d_a = cl::Buffer(context, CL_MEM_READ_ONLY, n*sizeof(double));
        d_b = cl::Buffer(context, CL_MEM_READ_ONLY, n*sizeof(double));
        d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, n*sizeof(double));

        // Bind memory buffers
        queue.enqueueWriteBuffer(d_a, CL_TRUE, 0, n*sizeof(double), h_a);
        queue.enqueueWriteBuffer(d_b, CL_TRUE, 0, n*sizeof(double), h_b);

        // Build kernel from source string
        cl::Program::Sources source(1,
            std::make_pair(kernelSource,strlen(kernelSource)));
        cl::Program program_ = cl::Program(context, source);
        program_.build(devices);

        // Create kernel object
        cl::Kernel kernel(program_, "vecAdd", &err);

        // bind kernel arguments to kernel
        kernel.setArg(0, d_a);
        kernel.setArg(1, d_b);
        kernel.setArg(2, d_c);
        kernel.setArg(3, n);

        // Number of work items in each work group
        cl::NDRange localSize(64);

        // Number of total work items -- localsize must be a divisor
        cl::NDRange globalSize((int)(ceil(n/(float)64)*64));

        // Enqueue kernel
        cl::Event event;
        queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            globalSize,
            localSize,
            NULL,
            &event
        );

        // Wait until kernel completion
        event.wait();

        // Read back d_c
        queue.enqueueReadBuffer(d_c, CL_TRUE, 0, n*sizeof(double), h_c);
    }
    catch(cl::Error err){
        std::cerr << "ERROR: "<<err.what()<<"("<<err.err()<<")"<<std::endl;
    }

    // Summing the arrays to make sure everything worked
    double sum = 0;
    for (int i = 0; i < n; ++i){
        sum += h_c[i];
    }

    std::cout << "final result is: " << sum << '\n';

    delete(h_a);
    delete(h_b);
    delete(h_c);
    
}

