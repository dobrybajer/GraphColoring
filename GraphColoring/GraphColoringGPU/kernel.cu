
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Graph.h"

#include <stdio.h>



cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int sizea,unsigned int sizeb,unsigned int sizec);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	
    int index = threadIdx.x;
	int count=0;
	for(int i =0;i<=index;i++)
		count+=a[i];

	for(int i =a[index];i>0;i--)
		c[count-i] = index;
}

int main()
{
	Graph graph = Graph();
	graph= Graph::ReadGraph("test.txt");

	for(int i = 0; i < graph._vertexCount; ++i,cout<<endl)
		for(int j = 0; j < graph.neighbourCount[i]; ++j)
				 cout<<graph.vertex[i][j]<<' '; 
	
	int neighCount=0;
	for(int i=0; i<graph._vertexCount; i++)
		neighCount+=graph.neighbourCount[i];

	int *tabCount=graph.neighbourCount;
	int *tabNeig=new int[neighCount];
	int *tabWyn=new int[neighCount];
	int tmp=0;
	for(int i= 0;i<graph._vertexCount;i++)
		for(int j=0;j<graph.neighbourCount[i];j++)
		{
			tabNeig[tmp]=graph.vertex[i][j];
			tmp++;
		}

		cudaError_t cudaStatus = addWithCuda(tabWyn, tabCount, tabNeig, graph._vertexCount,neighCount,neighCount);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	cout<<"Po przetworzeniu"<<endl;
	for(int i =0 ; i<neighCount;i++)
		 cout<<tabWyn[i]<<' '; 

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int sizea,unsigned int sizeb,unsigned int sizec)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, sizec * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, sizea * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, sizeb * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, sizea * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, sizeb * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, sizea>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, sizec * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
