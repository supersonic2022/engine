#include "imudata.h"

ImuData::ImuData(const double& gx, const double& gy, const double& gz,
	const double& ax, const double& ay, const double& az,
	const double& t) :
	_g(gx, gy, gz), _a(ax, ay, az), _t(t)
{
}

ImuData::ImuData(const Eigen::Matrix<double, 6, 1>& g_a, const double&& t) :
	_g(g_a.segment<3>(0)), _a(g_a.segment<3>(3)), _t(t)
{
	//std::cout << _t << std::endl;
}

ImgData::ImgData(const std::string img_dir, const double& t) : 
	_img_dir(img_dir), _t(t)
{

}
