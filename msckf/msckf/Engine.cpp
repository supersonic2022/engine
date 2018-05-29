#include "Engine.h"
#include "utility.h"


void Engine::init(BenchmarkNode* _benchmark)
{
	m_benchmark = _benchmark;
	ImuState::s_gyro_noise = _benchmark->dataset->imu_params[0].gyroscope_noise_density;
	ImuState::s_gyro_bias_noise = _benchmark->dataset->imu_params[0].gyroscope_random_walk;
	ImuState::s_acc_noise = _benchmark->dataset->imu_params[0].accelerometer_noise_density;
	ImuState::s_acc_bias_noise = _benchmark->dataset->imu_params[0].accelerometer_random_walk;

	m_P = Eigen::MatrixXd::Identity(15, 15);
	m_n = Matrix12d::Identity();
	m_n.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * ImuState::s_gyro_noise;
	m_n.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * ImuState::s_gyro_bias_noise;
	m_n.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * ImuState::s_acc_noise;
	m_n.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * ImuState::s_acc_bias_noise;

	m_R_C_I = _benchmark->dataset->cam_params[0].T_BS.block<3, 3>(0, 0);
	m_t_C_I = _benchmark->dataset->cam_params[0].T_BS.block<3, 1>(0, 3);
}

void Engine::predictState(ImuData _imuData)
{
	Matrix15d _F = F(m_imuState.m_q_I_G, _imuData._g, _imuData._a);
	Matrix15_12d _G = G(m_imuState.m_q_I_G);
	Matrix15d _Phi = Phi(_F, _imuData._t);
	Matrix15d _Q = _Phi * _G * m_n * _G.transpose() * _Phi.transpose() * _imuData._t;

	//predict state covariance
	m_P.block<15, 15>(0, 0) =
		_Phi * m_P.block<15, 15>(0, 0) * _Phi.transpose() + _Q;

	if (mv_camWindow.size() > 0) {
		m_P.block(0, 15, 15, m_P.cols() - 15) =
			_Phi * m_P.block(0, 15, 15, m_P.cols() - 15);
		m_P.block(15, 0, m_P.rows() - 15, 15) =
			m_P.block(15, 0, m_P.rows() - 15, 15) * _Phi.transpose();
	}

	//predict imu_state
	RK4(_imuData._g, _imuData._a, _imuData._t);
}

CamIDType Engine::augmentState()
{
	Eigen::MatrixXd _J = Eigen::MatrixXd(6, 15);
	_J.block<3, 3>(0, 0) = m_R_C_I;
	_J.block<3, 3>(3, 0) = Skew(m_R_C_I.transpose() * m_t_C_I);
	_J.block<3, 3>(3, 12) = Eigen::Matrix3d::Identity();

	size_t old_rows = m_P.rows();
	size_t old_cols = m_P.cols();
	m_P.conservativeResize(old_rows + 6, old_cols + 6);
	m_P.block(old_rows, 0, 6, old_cols + 6).setZero();
	m_P.block(0, old_cols, old_rows + 6, 6).setZero();

	m_P.block<6, 15>(old_rows, 0) = _J * m_P.block<15, 15>(0, 0);
	m_P.block<15, 6>(0, old_cols) = m_P.block<15, 15>(0, 0) * _J.transpose();
	m_P.block<6, 6>(old_rows, old_cols) = _J * m_P.block<15, 15>(0, 0) * _J.transpose();

	CamIDType camID = m_map.addCamState(quaternionMultiplication(q(m_R_C_I), m_imuState.m_q_I_G), m_imuState.m_p_G);
	mv_camWindow.push_back(camID);

	return camID;
}

void Engine::updateState()
{

}

void Engine::RK4(Eigen::Vector3d w, Eigen::Vector3d a, double dt)
{
	Eigen::Vector4d& q = m_imuState.m_q_I_G;
	Eigen::Vector3d& v = m_imuState.m_v_G;
	Eigen::Vector3d& p = m_imuState.m_p_G;

	//[1] P48
	double w_norm = w.norm();
	Eigen::Vector4d dq_dt, dq_dt2;
	if (w_norm > 1e-5) {
		dq_dt = (cos(w_norm*dt*0.5)*Eigen::Matrix4d::Identity() +
			1 / w_norm*sin(w_norm*dt*0.5)*Omega(w)) * q;
		dq_dt2 = (cos(w_norm*dt*0.25)*Eigen::Matrix4d::Identity() +
			1 / w_norm*sin(w_norm*dt*0.25)*Omega(w)) * q;
	}
	else {
		dq_dt = (Eigen::Matrix4d::Identity() + 0.5*dt*Omega(w)) *
			cos(w_norm*dt*0.5) * q;
		dq_dt2 = (Eigen::Matrix4d::Identity() + 0.25*dt*Omega(w)) *
			cos(w_norm*dt*0.25) * q;
	}
	Eigen::Matrix3d dR_dt_transpose = R(dq_dt).transpose();
	Eigen::Matrix3d dR_dt2_transpose = R(dq_dt2).transpose();

	// k1 = f(tn, yn)
	Eigen::Vector3d k1_v_dot = R(q).transpose()*a +
		ImuState::s_g_G;
	Eigen::Vector3d k1_p_dot = v;

	// k2 = f(tn+dt/2, yn+k1*dt/2)
	Eigen::Vector3d k1_v = v + k1_v_dot*dt / 2;
	Eigen::Vector3d k2_v_dot = dR_dt2_transpose*a +
		ImuState::s_g_G;
	Eigen::Vector3d k2_p_dot = k1_v;

	// k3 = f(tn+dt/2, yn+k2*dt/2)
	Eigen::Vector3d k2_v = v + k2_v_dot*dt / 2;
	Eigen::Vector3d k3_v_dot = dR_dt2_transpose*a +
		ImuState::s_g_G;
	Eigen::Vector3d k3_p_dot = k2_v;

	// k4 = f(tn+dt, yn+k3*dt)
	Eigen::Vector3d k3_v = v + k3_v_dot*dt;
	Eigen::Vector3d k4_v_dot = dR_dt_transpose*a +
		ImuState::s_g_G;
	Eigen::Vector3d k4_p_dot = k3_v;

	// yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
	q = dq_dt;
	q.normalize();
	v = v + dt / 6 * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot);
	p = p + dt / 6 * (k1_p_dot + 2 * k2_p_dot + 2 * k3_p_dot + k4_p_dot);
}

void Engine::process()
{
	while (m_benchmark->mq_maskQueue.size())
	{
		int flag = m_benchmark->mq_maskQueue.front();
		m_benchmark->mq_maskQueue.pop();

		if (flag == IMU)
		{
			ImuData t_imu = m_benchmark->mq_imuQueue.front();
			m_benchmark->mq_imuQueue.pop();

			predictState(t_imu);
		}
		if (flag == IMG)
		{

			ImgData t_img = m_benchmark->mq_imgQueue.front();
			m_benchmark->mq_imgQueue.pop();		
			
			CamIDType camID = augmentState();
			
			m_featManager.processImage(camID, t_img);

			

		}
	}
}
