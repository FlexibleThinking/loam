#include "loam_velodyne/BasicLaserMappingKernel.cuh"
#include "math_utils.h"
#include "loam_velodyne/nanoflann_pcl.h"

#include <stdio.h>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#define MAX_BLOCK 1024

CudaTest::CudaTest(void)
{
	printf("GPU instance created...\n\n");
}
CudaTest::~CudaTest(void)
{
	printf("GPU instance collapsed...\n\n");
}

/*__device__ void kernel_point_associate_to_map(const pcl::PointXYZI& pi, pcl::PointXYZI& po)
{
	po.x = pi.x;
	po.y = pi.y;
	po.intensity = pi.intensity;

	rotateZXY(po, _transformToBeMapped.rot_z, _transformToBeMapped.rot_x, _transformToMapped.rot_y);

	po.x += _transformToBeMapped.pos.x();
	po.y += _transformToBeMapped.pos.y();
	pos.z += _transformToBeMapped.pos.z();
}*/
__global__ void kernel_test(nanoflann::KdTreeFLANN<pcl::PointXYZI>& kdtreeCornerFromMap, nanoflann::KdTreeFLANN<pcl::PointXYZI>& kdtreeSurfFromMap)
{	
	printf("Gpu data size :: \n");
	//kernel_point_associate_to_map(pointOri, pointSel);

}
void CudaTest::initialize()
{
	matA0.setZero();
   	matB0.setConstant(-1);
   	matX0.setZero();

   	matA1.setZero();
   	matD1.setZero();
   	matV1.setZero();

	printf("Initializing Eigen Matrix... Done\n\n");
}
void CudaTest::process()
{   
	//This code would be in use after optimizeTransformToBeMapped() changed to gpu code 
	/*if (_laserCloudCornerFromMap->size() <= 10 || _laserCloudSurfFromMap->size() <= 100)
		 return;*/
	//printf("Memcpy test start\n");
	//printf("Original data size :: \n");
	
	nanoflann::KdTreeFLANN<pcl::PointXYZI> kernel_kd_tree_corner_from_map;
	nanoflann::KdTreeFLANN<pcl::PointXYZI> kernel_kd_tree_surf_from_map;
	printf("process started\n");
//	kernel_test<<<1, 1>>>(kdtreeCornerFromMap, kdtreeSurfFromMap);
	// TODO
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




