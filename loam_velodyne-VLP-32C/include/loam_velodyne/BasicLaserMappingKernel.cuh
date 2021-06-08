#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#if __CUDACC_VER_MAJOR__ >= 9
#undef __CUDACC_VER__
#define __CUDACC_VER__ \
	  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
#endif

#include "Twist.h"
//#include "CircularBuffer.h"
//#include <pcl/point_cloud.h>
//#include <pcl/point_types.h>
//#include <pcl/filters/voxel_grid.h>

#ifdef __cplusplus
extern "C"{
#endif
class CudaTest
{
	public:
		CudaTest(void);
		virtual ~CudaTest(void);
		int sum_cuda(int a, int b, int *c);
};
#ifdef __cplusplus
}
#endif
