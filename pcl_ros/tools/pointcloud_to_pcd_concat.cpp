/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id: pointcloud_to_pcd.cpp 33238 2010-03-11 00:46:58Z rusu $
 *
 */

// ROS core
#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>

// PCL includes
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Geometry>

using namespace std;

/**
\author Radu Bogdan Rusu

@b pointcloud_to_pcd is a simple node that retrieves a ROS point cloud message and saves it to disk into a PCD (Point
Cloud Data) file format.

**/
class PointCloudToPCD
{
  protected:
    ros::NodeHandle nh_;

  private:
    std::string prefix_;
    bool binary_;
    bool compressed_;
    std::string fixed_frame_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    pcl::PCLPointCloud2 _laserCloudSurround;

  public:
    string cloud_topic_;

    ros::Subscriber sub_;

    bool concatenate (pcl::PCLPointCloud2 &cloud1, const pcl::PCLPointCloud2 &cloud2)
    {
	if (cloud1.is_bigendian != cloud2.is_bigendian)
	  {
	    // In future, it might be possible to convert based on pcl::getFieldSize(fields.datatype)
	    PCL_ERROR ("[pcl::PCLPointCloud2::concatenate] Endianness of clouds does not match\n");
	    return (false);
	  }

	  const auto size1 = cloud1.width * cloud1.height;
	  const auto size2 = cloud2.width * cloud2.height;
	  //if one input cloud has no points, but the other input does, just select the cloud with points
	  switch ((bool (size1) << 1) + bool (size2))
	  {
	    case 1:
	      cloud1 = cloud2;
	    case 0:
	    case 2:
	      cloud1.header.stamp = std::max (cloud1.header.stamp, cloud2.header.stamp);
	      return (true);
	    default:
	      break;
	  }

	  // Ideally this should be in PCLPointField class since this is global behavior
	  auto field_eq = [](const auto& field1, const auto& field2)
	  {
	    // We're fine with the special RGB vs RGBA use case
	    return ((field1.name == field2.name) ||
		    (field1.name == "rgb" && field2.name == "rgba") ||
		    (field1.name == "rgba" && field2.name == "rgb"));
	  };

	  // A simple memcpy is possible if layout (name and order of fields) is same for both clouds
	  bool simple_layout = std::equal(cloud1.fields.begin (),
		                            cloud1.fields.end (),
		                            cloud2.fields.begin (),
		                            cloud2.fields.end (),
		                            field_eq);

	  struct FieldDetails
	  {
	    std::size_t idx1, idx2;
	    std::uint16_t size;
	    FieldDetails (std::size_t idx1_, std::size_t idx2_, std::uint16_t size_): idx1 (idx1_), idx2 (idx2_), size (size_)
	    {}
	  };
	  std::vector<FieldDetails> valid_fields;
	  const auto max_field_size = std::max (cloud1.fields.size (), cloud2.fields.size ());
	  valid_fields.reserve (max_field_size);

	  // @TODO: Refactor to return std::optional<std::vector<FieldDetails>>
	  // Store the details of fields with common data in both cloud, exit early if any errors are found
	  if (!simple_layout)
	  {
	    std::size_t i = 0, j = 0;
	    while (i < cloud1.fields.size () && j < cloud2.fields.size ())
	    {
	      if (cloud1.fields[i].name == "_")
	      {
		++i;
		continue;
	      }
	      if (cloud2.fields[j].name == "_")
	      {
		++j;
		continue;
	      }

	      if (field_eq(cloud1.fields[i], cloud2.fields[j]))
	      {
		// Assumption: cloud1.fields[i].datatype == cloud2.fields[j].datatype
		valid_fields.emplace_back(i, j, pcl::getFieldSize (cloud2.fields[j].datatype));
		++i;
		++j;
		continue;
	      }
	      PCL_ERROR ("[pcl::PCLPointCloud2::concatenate] Name of field %d in cloud1, %s, does not match name in cloud2, %s\n", i, cloud1.fields[i].name.c_str (), cloud2.fields[i].name.c_str ());
	      return (false);
	    }
	    // Both i and j should have exhausted their respective cloud.fields
	    if (i != cloud1.fields.size () || j != cloud2.fields.size ())
	    {
	      PCL_ERROR ("[pcl::PCLPointCloud2::concatenate] Number of fields to copy in cloud1 (%u) != Number of fields to copy in cloud2 (%u)\n", i, j);
	      return (false);
	    }
	  }

	  // Save the latest timestamp in the destination cloud
	  cloud1.header.stamp = std::max (cloud1.header.stamp, cloud2.header.stamp);

	  cloud1.is_dense = cloud1.is_dense && cloud2.is_dense;
	  cloud1.height = 1;
	  cloud1.width = size1 + size2;

	  if (simple_layout)
	  {
	    cloud1.data.insert (cloud1.data.end (), cloud2.data.begin (), cloud2.data.end ());
	    return (true);
	  }
	  const auto data1_size = cloud1.data.size ();
	  cloud1.data.resize(data1_size + cloud2.data.size ());
	  for (std::size_t cp = 0; cp < size2; ++cp)
	  {
	    for (const auto& field_data: valid_fields)
	    {
	      const auto& i = field_data.idx1;
	      const auto& j = field_data.idx2;
	      const auto& size = field_data.size;
	      // Leave the data for the skip fields untouched in cloud1
	      // Copy only the required data from cloud2 to the correct location for cloud1
	      memcpy (reinterpret_cast<char*> (&cloud1.data[data1_size + cp * cloud1.point_step + cloud1.fields[i].offset]),
		      reinterpret_cast<const char*> (&cloud2.data[cp * cloud2.point_step + cloud2.fields[j].offset]),
		      cloud2.fields[j].count * size);
	    }
	  }
	  return (true);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Callback
    void
      cloud_cb (const pcl::PCLPointCloud2::ConstPtr& cloud)
    {
      if ((cloud->width * cloud->height) == 0)
        return;
/*
      static uint64_t prevStamp = 0;

      if (cloud->header.stamp - prevStamp < 500000) {
        return;
      }

      prevStamp = cloud->header.stamp;
*/
      ROS_INFO ("Received %d data points in frame %s with the following fields: %s",
                (int)cloud->width * cloud->height,
                cloud->header.frame_id.c_str (),
                pcl::getFieldsList (*cloud).c_str ());

      Eigen::Vector4f v = Eigen::Vector4f::Zero ();
      Eigen::Quaternionf q = Eigen::Quaternionf::Identity ();
      if (!fixed_frame_.empty ()) {
        if (!tf_buffer_.canTransform (fixed_frame_, cloud->header.frame_id, pcl_conversions::fromPCL (cloud->header.stamp), ros::Duration (3.0))) {
          ROS_WARN("Could not get transform!");
          return;
        }

        Eigen::Affine3d transform;
        transform = tf2::transformToEigen (tf_buffer_.lookupTransform (fixed_frame_, cloud->header.frame_id,  pcl_conversions::fromPCL (cloud->header.stamp)));
        v = Eigen::Vector4f::Zero ();
        v.head<3> () = transform.translation ().cast<float> ();
        q = transform.rotation ().cast<float> ();
      }

      concatenate(_laserCloudSurround, *cloud);

      std::stringstream ss;
      ss << prefix_ << "concat_test"/*cloud->header.stamp*/ << ".pcd";
      ROS_INFO ("Data saved to %s", ss.str ().c_str ());

      pcl::PCDWriter writer;
      if(binary_)
	{
	  if(compressed_)
	    {
	      writer.writeBinaryCompressed (ss.str (), _laserCloudSurround/**cloud*/, v, q);
	    }
	  else
	    {
	      writer.writeBinary (ss.str (), _laserCloudSurround/**cloud*/, v, q);
	    }
	}
      else
	{
	  writer.writeASCII (ss.str (), _laserCloudSurround/**cloud*/, v, q, 8);
	}

    }

    ////////////////////////////////////////////////////////////////////////////////
    PointCloudToPCD () : binary_(false), compressed_(false), tf_listener_(tf_buffer_)
    {
      // Check if a prefix parameter is defined for output file names.
      ros::NodeHandle priv_nh("~");
      if (priv_nh.getParam ("prefix", prefix_))
        {
          ROS_INFO_STREAM ("PCD file prefix is: " << prefix_);
        }
      else if (nh_.getParam ("prefix", prefix_))
        {
          ROS_WARN_STREAM ("Non-private PCD prefix parameter is DEPRECATED: "
                           << prefix_);
        }

      priv_nh.getParam ("fixed_frame", fixed_frame_);
      priv_nh.getParam ("binary", binary_);
      priv_nh.getParam ("compressed", compressed_);
      if(binary_)
	{
	  if(compressed_)
	    {
	      ROS_INFO_STREAM ("Saving as binary compressed PCD");
	    }
	  else
	    {
	      ROS_INFO_STREAM ("Saving as binary PCD");
	    }
	}
      else
	{
	  ROS_INFO_STREAM ("Saving as binary PCD");
	}

      cloud_topic_ = "input";

      sub_ = nh_.subscribe (cloud_topic_, 1,  &PointCloudToPCD::cloud_cb, this);
      ROS_INFO ("Listening for incoming data on topic %s",
                nh_.resolveName (cloud_topic_).c_str ());
    }    
};

/* ---[ */
int
  main (int argc, char** argv)
{
  ros::init (argc, argv, "pointcloud_to_pcd_concat", ros::init_options::AnonymousName);

  PointCloudToPCD b;
  ros::spin ();

  return (0);
}
/* ]--- */
