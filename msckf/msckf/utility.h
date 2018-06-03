#pragma once

#include "global.h"

Eigen::Matrix3d Skew(Eigen::Vector3d w);

Eigen::Matrix4d Omega(Eigen::Vector3d w);

Eigen::Matrix3d R(Eigen::Vector4d q);

Eigen::Vector4d q(Eigen::Matrix3d R);

Matrix15d F(Eigen::Vector4d q, Eigen::Vector3d w, Eigen::Vector3d a);

Matrix15_12d G(Eigen::Vector4d q);

Matrix15d Phi(Matrix15d F, double dtime);

Eigen::Vector4d quaternionMultiplication(
	const Eigen::Vector4d& q1,
	const Eigen::Vector4d& q2);

Eigen::Vector4d smallAngleQuaternion(
	const Eigen::Vector3d& dtheta);




