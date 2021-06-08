#include "loam_velodyne/BasicLaserMappingKernel.cuh"
#include <stdio.h>
CudaTest::CudaTest(void)
{
}
CudaTest::~CudaTest(void)
{
}

__global__ void sum_kernel(int a, int b, int *c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	c[tid] = a + b;
}
int CudaTest::sum_cuda(int a, int b, int *c)
{
	int *f;
	cudaMalloc((void**)&f, sizeof(int)* 1);
	cudaMemcpy(f, c, sizeof(int)* 1, cudaMemcpyHostToDevice);

	sum_kernel <<<1, 1>>>(a, b, f);

	cudaMemcpy(c, f, sizeof(int) *1, cudaMemcpyDeviceToHost);

	cudaFree(f);

	return true;
}
