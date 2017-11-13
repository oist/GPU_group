__global__ void reduce(const unsigned int* d_data,
                        const size_t sz, unsigned int * d_out)
 {
   //  Indexing will be: sz * binId + threadIdx.x
   extern __shared__ unsigned int sh_hist [];
   int binId = blockIdx.x;
   int myId = threadIdx.x + sz * binId;
   int tId = threadIdx.x;
   if ( tId >= sz)
   {
     sh_hist[tId] = 0;
   }
   else
   // Fill in the shared memory
     sh_hist[tId] = d_histo[myId];
   __syncthreads();
   for  (unsigned int s = blockDim.x /2; s > 0; s >>=1)
   {
     if (tId < s)
     {
       sh_hist[tId] += sh_hist[tId+s];
     }
     __syncthreads();
   }
   if (tId == 0)
   {
     d_out[blockIdx.x] = sh_hist[0];
     printf("%i : %i\n", blockIdx.x, sh_hist[0]);
   }
 }