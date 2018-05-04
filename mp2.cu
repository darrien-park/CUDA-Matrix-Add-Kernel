//Darrien Park

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#define BLOCK_SIZE 16

const size_t w = 200;

//kernel functions to be called from the host and executed in the gpu
//produces one output matrix element per thread
__global__ void MatrixAddKernel(float* a, float *b, float *sum, int width){

	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if (row < width && col < width){					//only threads within range
			sum[row*width + col] = a[row*width + col] + b[row*width + col];
	}
}

//produces one output matrix row per thread
__global__ void MatrixAddRow(float* a, float *b, float *sum, int width){

	int row = blockIdx.x*blockDim.x + threadIdx.x;

	if (row < width){				//only threads within range
		int j;
		for (j = 0; j < width; j++)
			sum[row*width + j] = a[row*width + j] + b[row*width + j];
	}
}

//produces one output matrix row per thread
__global__ void MatrixAddCol(float* a, float *b, float *sum, int width){

	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if (col < width){				//only threads within range
		for (int i = 0; i < width; i++)
			sum[col + width*i] = a[col + width* i] + b[col + width * i];
	}

}

//define a new type for matrix so that matrices can be stored on the heap; execution will not crash on large matrix sizes
typedef float squareMatrix[w];

//function to check if the resultant matrix from the CPU is the same as the GPU
void correct_output(squareMatrix *CPUout, squareMatrix *GPUout, int width){
	for (int i = 0; i < width; i++)
		for (int j = 0; j < width; j++){
			if (CPUout[i][j] != GPUout[i][j])
				printf("TEST FAILED\n");
		}
	printf("TEST PASSED\n");
}

int main(){
	//define and initialize variables, allocate memory in heap for matrices
	int size = w*w*sizeof(float);
	squareMatrix *a, *b, *GPUsum, *CPUsum;
	a		= (squareMatrix *)malloc(size);
	b		= (squareMatrix *)malloc(size);
	GPUsum	= (squareMatrix *)malloc(size);
	CPUsum	= (squareMatrix *)malloc(size);

	//populate the matrix with randum numbers between 0 and 10 to read output easily
	srand(time(NULL));
	for(int i =0; i<w; i++)
		for(int j=0;j<w;j++){
			a[i][j] = rand() % (10 + 1 - 0) + 0;
			b[i][j] = rand() % (10 + 1 - 0) + 0;
		}

	//find number of blocks required which is width = width of matrix/block size
	int NumBlocks = w/BLOCK_SIZE;
	//if remainder, extra block is needed
	if(w % BLOCK_SIZE) NumBlocks++;
	//set grid dimensions
	dim3 dimGrid(NumBlocks,NumBlocks);	//for 16x16 parameters are (NumBlocks, Numblocks), for 16 block/thread (Numblocks)
	//set block dimensions 16x16
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);	//for 16x16 (BLOCK_SIZE, BLOCK_SIZE)


	float *d_a, *d_b, *d_sum;
	//allocate host memory onto device
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_sum, size);

	cudaMemcpyAsync(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_b, b, size, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_sum, GPUsum, size, cudaMemcpyHostToDevice);

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaDeviceSynchronize();
	float gpu_time = 0.0f;

	//record gpu calculation time
	cudaEventRecord(start,0);
	MatrixAddKernel<<<dimGrid,dimBlock>>>(d_a, d_b, d_sum, w);	//change kernel name to compare performance
	cudaEventRecord(stop,0);

	cudaMemcpy(GPUsum, d_sum, size, cudaMemcpyDeviceToHost);

	//CPU calculation
	cudaEvent_t CPUstart, CPUstop;
	cudaEventCreate(&CPUstart);
	cudaEventCreate(&CPUstop);
	cudaEventRecord(CPUstart);
	for(int i = 0; i < w; i++)
		for(int j =0; j<w; j++){
			CPUsum[i][j] = a[i][j]+b[i][j];
	}
	cudaEventRecord(CPUstop);
	cudaEventSynchronize(CPUstop);
	float cpu_time = 0.0f;
	cudaEventElapsedTime(&cpu_time, CPUstart, CPUstop);
	printf("Time spent executing bv the CPU: %.2f\n",cpu_time);

	unsigned long int counter = 0;
	while(cudaEventQuery(stop) == cudaErrorNotReady){
		counter ++;
	}

	cudaEventElapsedTime(&gpu_time,start,stop);

	printf("Time spent executing bv the GPU: %.2f\n",gpu_time);

	correct_output(CPUsum, GPUsum, w);

	//free memory space pointed to by host
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(GPUsum);
	//free memory space pointed to by device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_sum);
	cudaDeviceReset();

	return 0;
}
