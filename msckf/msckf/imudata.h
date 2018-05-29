#pragma once

#include "global.h"

class ImuData
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	ImuData(const double& gx, const double& gy, const double& gz,
		const double& ax, const double& ay, const double& az,
		const double& t);
	//IMUData(const IMUData& imu);

	ImuData(const Eigen::Matrix<double, 6, 1>& g_a,const double&& t);

	// Raw data of imu's
	Eigen::Vector3d _g;    //gyr data
	Eigen::Vector3d _a;    //acc data
	double _t;      //time duration
};

class ImgData
{
public:
	ImgData(const std::string img_dir, const double& t);

	std::string _img_dir;
	double _t;
};