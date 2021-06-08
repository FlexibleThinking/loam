#include "loam_velodyne/BasicLaserMappingKernel.cuh"
#include "math_utils.h"
#include "loam_velodyne/nanoflann_pcl.h"

#include <stdio.h>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <time.h>
#define MAX_BLOCK 255
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
__global__ void kernel_test(nanoflann::KdTreeFLANN<pcl::PointXYZI> kdtreeCornerFromMap, nanoflann::KdTreeFLANN<pcl::PointXYZI> kdtreeSurfFromMap)
{

}
void CudaTest::initialize()
{
	matA0.setZero();
   	matB0.setConstant(-1);
   	matX0.setZero();

   	matA1.setZero();
   	matD1.setZero();
   	matV1.setZero();
}
void CudaTest::process()
{
	// write kernel code
}
/*int CudaTest::sum_cuda(int a, int b, int *c)
{
	int *f;
	cudaMalloc((void**)&f, sizeof(int)* 1);
	cudaMemcpy(f, c, sizeof(int)* 1, cudaMemcpyHostToDevice);

	sum_kernel <<<1, 1>>>(a, b, f);

	cudaMemcpy(c, f, sizeof(int) *1, cudaMemcpyDeviceToHost);

	cudaFree(f);

	return true;
}*/
void CudaTest::set_laser_cloud_corner_from_map(pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerFromMapFromHost){
	kdtreeCornerFromMap.setInputCloud(laserCloudCornerFromMapFromHost);
}
void CudaTest::set_laser_cloud_surf_from_map(pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfFromMapFromHost){
	kdtreeSurfFromMap.setInputCloud(laserCloudSurfFromMapFromHost);
}
