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

	void updateState();

	void process();

private:
	void RK4(Eigen::Vector3d w, Eigen::Vector3d a, double dt);

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