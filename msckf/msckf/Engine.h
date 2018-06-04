#pragma once

#include "map.h"
#include "imudata.h"
#include "benchmark.h"
#include "FeatureManager.h"

class Engine
{
public:
	Engine():m_featManager(&m_map){}
	~Engine() {}

public:
	void init(BenchmarkNode* _benchmark);

	void predictState(ImuData _imuData);

	CamIDType augmentState();

	void updateState(int jacobian_row_size);

	void process();



private:
	void RK4(
		Eigen::Vector3d w, 
		Eigen::Vector3d a, 
		double dt);

	void featureJacobian(
		const FeatIDType& featID,
		const std::vector<CamIDType>& camIDList,
		Eigen::MatrixXd& H_x, 
		Eigen::VectorXd& r);

	void measurementJacobian(
		const CamIDType& camID,
		const FeatIDType& featID,
		Eigen::Matrix<double, 2, 6>& H_x,
		Eigen::Matrix<double, 2, 3>& H_f,
		Eigen::Vector2d& r);

	//need add chi square table
	bool gatingTest(
		const Eigen::MatrixXd& H, 
		const Eigen::VectorXd& r, 
		const int& dof) {

		//Eigen::MatrixXd P1 = H * m_P * H.transpose();
		//Eigen::MatrixXd P2 = OBSERVATION_NOISE *
		//	Eigen::MatrixXd::Identity(H.rows(), H.rows());
		//double gamma = r.transpose() * (P1 + P2).ldlt().solve(r);

		//if (gamma < chi_squared_test_table[dof]) {
		//	//cout << "passed" << endl;
		//	return true;
		//}
		//else {
		//	cout << "failed" << endl;
		//	return false;
		//}

		return true;
	}

	void measurementUpdate(
		const Eigen::MatrixXd& H, 
		const Eigen::VectorXd& r);


	void findRedundantCamStates(
		std::vector<CamIDType>& camIDList);

	void pruneCamStateBuffer();

	//global map
	Map m_map;
	//camera state in sliding window
	std::vector<CamIDType> mv_camWindow;
	//original noise matrix
	Matrix12d m_n;
	//rotation from Imu frame to camera frame
	Eigen::Matrix3d m_R_C_I;
	//translation from Imu frame to camera frame
	Eigen::Vector3d m_t_C_I;
	//current imu state
	ImuState m_imuState;
	//current covariance
	Eigen::MatrixXd m_P;

	BenchmarkNode* m_benchmark;
	FeatureManager m_featManager;

};