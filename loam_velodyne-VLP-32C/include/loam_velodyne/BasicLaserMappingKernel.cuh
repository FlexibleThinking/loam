#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#if __CUDACC_VER_MAJOR__ >= 9
#undef __CUDACC_VER__
#define __CUDACC_VER__ \
	  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
#endif

#include "Twist.h"
#include "CircularBuffer.h"
#include "nanoflann_pcl.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#ifdef __basic_laser_mapping__kernel
extern "C"{
#endif
class CudaTest
{
	public:
		CudaTest(void);
		virtual ~CudaTest(void);

		void set_laser_cloud_corner_from_map(pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerFromMapFromHost);
		void set_laser_cloud_surf_from_map(pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfFromMapFromHost);
		void set_corner_stack_down_size(pcl::PointCloud<pcl::PointXYZI>::Ptr corner_from_host);
		void set_surf_stack_down_size(pcl::PointCloud<pcl::PointXYZI>::Ptr surf_from_host);
		void set_max_iter(size_t max_iter);

		void initialize();
		void process();

	private:
		pcl::PointXYZI pointsel, pointOri, coeff;

		nanoflann::KdTreeFLANN<pcl::PointXYZI> kdtreeCornerFromMap;
		nanoflann::KdTreeFLANN<pcl::PointXYZI> kdtreeSurfFromMap;

		Eigen::Matrix<float, 5, 3> matA0;
   		Eigen::Matrix<float, 5, 1> matB0;
		Eigen::Vector3f matX0;
   		Eigen::Matrix3f matA1;
		Eigen::Matrix<float, 1, 3> matD1;
  		Eigen::Matrix3f matV1;
		
		pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudCornerStackDS;
		pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudSurfStackDS;

		size_t laserCloudCornerStackNum = 0;
		size_t laserCloudSurfStackNum = 0;
		
		bool isDegenerate = false;
 		Eigen::Matrix<float, 6, 6> matP;

		size_t _maxIterations;
		
};
#ifdef __basic_laser_mapping__kernel
}
#endif
