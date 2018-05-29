#include "utility.h"

Eigen::Matrix3d Skew(Eigen::Vector3d w)
{
	Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
	skew(1, 2) = -w(0);
	skew(2, 1) = w(0);
	skew(0, 2) = w(1);
	skew(2, 0) = -w(1);
	skew(0, 1) = -w(2);
	skew(1, 0) = w(2);
	return skew;
}

Eigen::Matrix4d Omega(Eigen::Vector3d w)
{
	Eigen::Matrix4d omega = Eigen::Matrix4d::Zero();
	omega.block<3, 3>(0, 0) = Skew(w);
	omega.block<3, 1>(0, 3) = w;
	omega.block<1, 3>(3, 0) = w.transpose();
	return omega;
}

//[1](117)
Eigen::Matrix3d R(Eigen::Vector4d q)
{
	Eigen::Matrix3d _R = Eigen::Matrix3d::Zero();
	Eigen::Vector3d qv = q.segment(0, 3);
	double qw = q(3);

	_R = (qw * qw - qv.transpose() * qv) * Eigen::Matrix3d::Identity() +
		2 * qv * qv.transpose() +
		2 * qw * Skew(qv);

	return _R;
}

//[3](96~99)
Eigen::Vector4d q(Eigen::Matrix3d R)
{
	Eigen::Vector4d _q = Eigen::Vector4d::Zero();

	Eigen::Vector4d score;
	score(0) = R(0, 0);
	score(1) = R(1, 1);
	score(2) = R(2, 2);
	score(3) = R.trace();

	int max_row = 0, max_col = 0;
	score.maxCoeff(&max_row, &max_col);

	if (max_row == 0) {
		_q(0) = std::sqrt(1 + 2 * R(0, 0) - R.trace()) / 2.0;
		_q(1) = (R(0, 1) + R(1, 0)) / (4 * _q(0));
		_q(2) = (R(0, 2) + R(2, 0)) / (4 * _q(0));
		_q(3) = (R(1, 2) - R(2, 1)) / (4 * _q(0));
	}
	else if (max_row == 1) {
		_q(1) = std::sqrt(1 + 2 * R(1, 1) - R.trace()) / 2.0;
		_q(0) = (R(0, 1) + R(1, 0)) / (4 * _q(1));
		_q(2) = (R(1, 2) + R(2, 1)) / (4 * _q(1));
		_q(3) = (R(2, 0) - R(0, 2)) / (4 * _q(1));
	}
	else if (max_row == 2) {
		_q(2) = std::sqrt(1 + 2 * R(2, 2) - R.trace()) / 2.0;
		_q(0) = (R(0, 2) + R(2, 0)) / (4 * _q(2));
		_q(1) = (R(1, 2) + R(2, 1)) / (4 * _q(2));
		_q(3) = (R(0, 1) - R(1, 0)) / (4 * _q(2));
	}
	else {
		_q(3) = std::sqrt(1 + R.trace()) / 2.0;
		_q(0) = (R(1, 2) - R(2, 1)) / (4 * _q(3));
		_q(1) = (R(2, 0) - R(0, 2)) / (4 * _q(3));
		_q(2) = (R(0, 1) - R(1, 0)) / (4 * _q(3));
	}

	if (_q(3) < 0) _q = -_q;
	_q.normalize();

	return _q;
}

//linear continuous time model for imu error state propogation
Matrix15d F(Eigen::Vector4d q, Eigen::Vector3d w, Eigen::Vector3d a)
{
	Matrix15d _F = Matrix15d::Zero();
	_F.block<3, 3>(0, 0) = -Skew(w);
	_F.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
	_F.block<3, 3>(6, 0) = -R(q).transpose() * Skew(a);
	_F.block<3, 3>(6, 9) = -R(q).transpose();
	_F.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();
	return _F;
}

Matrix15_12d G(Eigen::Vector4d q)
{
	Matrix15_12d _G = Matrix15_12d::Zero();
	_G.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
	_G.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
	_G.block<3, 3>(6, 6) = -R(q).transpose();
	_G.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
	return _G;
}

//transition matrix (3 orfer Taylor approximation)
Matrix15d Phi(Matrix15d F, double dtime)
{
	Matrix15d Fdt = F * dtime;
	Matrix15d Fdt_square = Fdt * Fdt;
	Matrix15d Fdt_cube = Fdt_square * Fdt;
	Matrix15d _Phi = Matrix15d::Identity() +
		Fdt + 0.5*Fdt_square + (1.0 / 6.0)*Fdt_cube;
	return _Phi;
}

Eigen::Vector4d quaternionMultiplication(
	const Eigen::Vector4d& q1,
	const Eigen::Vector4d& q2) {
	Eigen::Matrix4d L;
	L(0, 0) = q1(3); L(0, 1) = q1(2); L(0, 2) = -q1(1); L(0, 3) = q1(0);
	L(1, 0) = -q1(2); L(1, 1) = q1(3); L(1, 2) = q1(0); L(1, 3) = q1(1);
	L(2, 0) = q1(1); L(2, 1) = -q1(0); L(2, 2) = q1(3); L(2, 3) = q1(2);
	L(3, 0) = -q1(0); L(3, 1) = -q1(1); L(3, 2) = -q1(2); L(3, 3) = q1(3);

	Eigen::Vector4d q = L * q2;
	q.normalize();
	return q;
}




