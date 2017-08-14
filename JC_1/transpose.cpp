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
int *test_copy(int *array, int width, int height);

// Function to test the transpose kernel
int *test_transpose(int *array, int width, int height);

// Function to print an array
void print_array(int *array, int width, int height);

// Function to open a file and return a const char *
const char *open_file(std::string filename);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

// main function
int main(){

    // Creating 32x32 array to mess around with
    int array[TILE*TILE];
    for (int i = 0; i < TILE*TILE; i++){
        array[i] = i % 2;
    }

    std::cout << "ORIGINAL ARRAY:\n";
    print_array(array, TILE, TILE);

    std::cout << "\n\n";

    int *copied_array = test_copy(array, TILE, TILE);
    std::cout << "COPIED ARRAY:\n";
    print_array(copied_array, TILE, TILE);
    std::cout << "\n\n";

    int *transposed_array = test_transpose(array, TILE, TILE);
    std::cout << "TRANSPOSED ARRAY:\n";
    print_array(transposed_array, TILE, TILE);
    std::cout << "\n\n";
    
}


/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to test the copy kernel
int *test_copy(int *array, int width, int height){

    int *out;
    out = new int[width*height];

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
                          width*height*sizeof(int));
        d_out = cl::Buffer(context, CL_MEM_WRITE_ONLY, 
                           width*height*sizeof(int));

        // Bind memory buffer
        queue.enqueueWriteBuffer(d_in, CL_TRUE, 0, width*height*sizeof(int),
                                 array);

        // now to read in the kernel
        const char *copy_kernel = open_file("copy2d.cl");

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
        cl::NDRange localSize(TILE, BLOCK);

        cl::NDRange globalSize(width,height);

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

        queue.enqueueReadBuffer(d_out, CL_TRUE, 0, width*height*sizeof(int),
                                out);

        
    }
    catch(cl::Error err){
        std::cerr << "ERROR: " << err.what() 
                  << "("<<err.err()<<")" << std::endl;

    }

    return out;
}

// Function to test the transpose kernel
int *test_transpose(int *array, int width, int height){
    int *out;
    out = new int[width*height];

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
                          width*height*sizeof(int));
        d_out = cl::Buffer(context, CL_MEM_WRITE_ONLY, 
                           width*height*sizeof(int));

        // Bind memory buffer
        queue.enqueueWriteBuffer(d_in, CL_TRUE, 0, width*height*sizeof(int),
                                 array);

        // now to read in the kernel
        const char *trans_kernel = open_file("transpose2d.cl");

        cl::Program::Sources source(1, 
            std::make_pair(trans_kernel, strlen(trans_kernel)));

        cl::Program program = cl::Program(context, source);
        err = program.build(devices);
        if (err != CL_SUCCESS){
            std::cout << "you got problems" << '\n';
        }
        cl::Kernel kernel(program, "transpose2d", &err);

        // Bind arguments for kernel
        kernel.setArg(0,d_in);
        kernel.setArg(1,d_out);
        kernel.setArg(2,TILE*(TILE)*sizeof(int), NULL);
        kernel.setArg(3,width);
        kernel.setArg(4,height);
        kernel.setArg(5,TILE);
        kernel.setArg(6,BLOCK);

        // Number of work-items
        cl::NDRange localSize(TILE, BLOCK);

        cl::NDRange globalSize(width, height);

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

        queue.enqueueReadBuffer(d_out, CL_TRUE, 0, width*height*sizeof(int),
                                out);

        
    }
    catch(cl::Error err){
        std::cerr << "ERROR: " << err.what() 
                  << "("<<err.err()<<")" << std::endl;

    }

    return out;

}

// Function to print an array
void print_array(int *array, int width, int height){

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
