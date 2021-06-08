#include "loam_velodyne/BasicLaserMappingKernel.cuh"
#include "math_utils.h"
#include "loam_velodyne/nanoflann_pcl.h"

#include <stdio.h>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#define MAX_BLOCK 1024

CudaTest::CudaTest(void)
{
}
CudaTest::~CudaTest(void)
{
}

__global__ void kernel_test(nanoflann::KdTreeFLANN<pcl::PointXYZI> kdtreeCornerFromMap, nanoflann::KdTreeFLANN<pcl::PointXYZI> kdtreeSurfFromMap)
{	// TODO
	// write kernel code
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
{   // TODO
	// memcpy to device
	// memcpy to host
}
void CudaTest::set_laser_cloud_corner_from_map(pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerFromMapFromHost){
	kdtreeCornerFromMap.setInputCloud(laserCloudCornerFromMapFromHost);
}
void CudaTest::set_laser_cloud_surf_from_map(pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfFromMapFromHost){
	kdtreeSurfFromMap.setInputCloud(laserCloudSurfFromMapFromHost);
}

void CudaTest::set_corner_stack_down_size(pcl::PointCloud<pcl::PointXYZI>::Ptr corner_from_host)
{
	_laserCloudCornerStackDS = corner_from_host;
	laserCloudCornerStackNum = _laserCloudCornerStackDS->size();
}
void CudaTest::set_surf_stack_down_size(pcl::PointCloud<pcl::PointXYZI>::Ptr surf_from_host)
{
	_laserCloudSurfStackDS = surf_from_host;
	laserCloudSurfStackNum = _laserCloudSurfStackDS->size();
}

void CudaTest::set_max_iter(size_t max_iter)
{
	_maxIterations = max_iter;
}




