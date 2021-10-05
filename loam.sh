#!/bin/bash
#`apt-get update -y`
#`apt-get upgrade -y`
#echo `apt-get install -y vim`

echo "===================== Build essential ========================" 

echo "check apt-utils ..."
`dpkg -s apt-utils > /dev/null 2>&1`
if [[ $? -ne 0 ]]; then
	echo "install apt-utils ..."
	echo `apt-get install -y apt-utils`
fi
echo -e "\033[32m>> Done ...\033[0m\n"

echo "check clang ..."
`dpkg -s clang > /dev/null 2>&1`
if [[ $? -ne 0 ]]; then
	echo `apt-get install -y clang`
fi
echo -e "\033[32m>> Done ...\033[0m\n"

echo "check gcc..."
`dpkg -s gcc > /dev/null 2>&1`
if [[ $? -ne 0 ]]; then
	echo `apt-get install -y gcc`
fi	
echo -e "\033[32m>> Done ...\033[0m\n"

echo "check g++..."
`dpkg -s g++ > /dev/null 2>&1`
if [[ $? -ne 0 ]]; then
	echo `apt-get install -y g++`
fi
echo -e "\033[32m>> Done ...\033[0m\n"

echo "check build essential"
`dpkg -s build-essential > /dev/null 2>&1`
if [[ $? -ne 0 ]]; then
	echo `apt-get install -y build-essential`
fi
echo -e "\033[32m>> Done ...\033[0m\n"

#echo -e "\033[33m>> [Warning]    File name change $O_F/$i --> $O_F/${i//fastq.gz*/fastq.gz}\033[0m\n"
echo "check wget ..."
`dpkg -s wget > /dev/null 2>&1`
if [[ $? -ne 0 ]]; then
	echo `apt-get install wget`
fi
echo -e "\033[32m>> Done ...\033[0m\n"

echo "check Openssl..."
`dpkg -s libssl-dev > /dev/null 2>&1`
if [[ $? -ne 0 ]]; then
	echo `apt-get install -y libssl-dev`
fi
echo -e "\033[32m>> Done ...\033[0m\n"
echo "===================================="

echo "check pcap lib"
#echo `apt-get -y install libpcap-dev`
echo -e "\033[32m>> Done ...\033[0m\n"

echo "install cmake ..."
`dpkg -s cmake > /dev/null 2>&1`
if [[ $? -eq 0 ]]; then
	echo `apt purge cmake`
fi
FIND_CMAKE=`find ./ -name "cmake-3.19.2"`
if [[ $? -ne 0 ]]; then
	echo `wget https://github.com/Kitware/Cmake/releases/download/v3.19.2/cmake-3.19.2.tar.gz`
	echo `tar zxf cmake-3.19.2.tar.gz`
fi
echo `cd /cmake-3.19.2 && ./bootstrap --prefix=/usr && make && make install`
if [[ $? -ne 0 ]];then
	echo "\033[33m>> [Warning]    failed\033[0m\n"
	exit 1
fi
echo -e "\033[32m>> Done ...\033[0m\n"

echo "install pcl lib ..."
echo `wget https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-1.12.0.tar.gz`
echo `tar xvfz pcl-1.12.0.tar.gz`
echo `cd pcl-pcl-1.12.0 && mkdir build && cd build`
echo `cmake ..`
echo `make -j4 install`
echo `dpkg -s libpcl-dev | grep 'Version'`
echo `dpkg -s libboost-dev | grep 'Version' pkg-config --modversion eigen3`
echo -e "\033[32m>> Done ...\033[0m\n"
#%%%%%%%%%%%%%%%%%%%%%%%%%%
## ROS

echo "ROS install -> https://varhowto.com/install-ros-noetic-ubuntu-20-04/ " 
