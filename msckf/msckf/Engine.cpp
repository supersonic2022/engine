#include "Engine.h"
#include "utility.h"
//#include <Eigen/SPQRSupport>


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

void Engine::updateState(int jacobian_row_size)
{
	if (m_featManager.valideTracks.size() == 0)
		return;
	
	Eigen::MatrixXd H_x = Eigen::MatrixXd::Zero(jacobian_row_size,
		15 + 6 * mv_camWindow.size());
	Eigen::VectorXd r = Eigen::VectorXd::Zero(jacobian_row_size);
	int stack_cntr = 0;

	for (auto featID: m_featManager.valideTracks)
	{
		std::vector<CamIDType> camIDList;
		m_map.getCamStateList(featID, camIDList);

		Eigen::MatrixXd H_xj;
		Eigen::VectorXd r_j;
		featureJacobian(featID, camIDList, H_xj, r_j);

		if (gatingTest(H_xj, r_j, camIDList.size() - 1)) {
			H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
			r.segment(stack_cntr, r_j.rows()) = r_j;
			stack_cntr += H_xj.rows();
		}

		// Put an upper bound on the row size of measurement Jacobian,
		// which helps guarantee the executation time.
		//if (stack_cntr > 1500) break;
	}

	H_x.conservativeResize(stack_cntr, H_x.cols());
	r.conservativeResize(stack_cntr);

	// Perform the measurement update step.
	measurementUpdate(H_x, r);

	// Remove all processed features from the map.
	m_featManager.removeInvalid();
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
		
			int jacobian_row_size = m_featManager.processImage(camID, t_img);

			updateState(jacobian_row_size);

			pruneCamStateBuffer();
		}
	}
}

void Engine::featureJacobian(
	const FeatIDType& featID,
	const std::vector<CamIDType>& camIDList,
	Eigen::MatrixXd& H_x, Eigen::VectorXd& r)
{
	FeatState* featState = m_map.getFeatState(featID);

	int jacobian_row_size = 0;
	jacobian_row_size = 2 * camIDList.size();

	Eigen::MatrixXd H_xj = Eigen::MatrixXd::Zero(jacobian_row_size,
		15 + mv_camWindow.size() * 6);
	Eigen::MatrixXd H_fj = Eigen::MatrixXd::Zero(jacobian_row_size, 3);
	Eigen::VectorXd r_j = Eigen::VectorXd::Zero(jacobian_row_size);
	int stack_cntr = 0;

	for (int i = 0; i < camIDList.size(); i++) {

		CamIDType camID = camIDList[i];
		Eigen::Matrix<double, 2, 6> H_xi = Eigen::Matrix<double, 2, 6>::Zero();
		Eigen::Matrix<double, 2, 3> H_fi = Eigen::Matrix<double, 2, 3>::Zero();
		Eigen::Vector2d r_i = Eigen::Vector2d::Zero();
		measurementJacobian(camID, featID, H_xi, H_fi, r_i);

		// Stack the Jacobians.
		H_xj.block<2, 6>(stack_cntr, 15 + 6 * i) = H_xi;
		H_fj.block<2, 3>(stack_cntr, 0) = H_fi;
		r_j.segment<2>(stack_cntr) = r_i;
		stack_cntr += 2;
	}

	// Project the residual and Jacobians onto the nullspace
	// of H_fj.
	Eigen::JacobiSVD<Eigen::MatrixXd> svd_helper(H_fj, Eigen::ComputeFullU | Eigen::ComputeThinV);
	Eigen::MatrixXd A = svd_helper.matrixU().rightCols(
		jacobian_row_size - 3);

	H_x = A.transpose() * H_xj;
	r = A.transpose() * r_j;

	return;
}

void Engine::measurementJacobian(
	const CamIDType& camID,
	const FeatIDType& featID,
	Eigen::Matrix<double, 2, 6>& H_x, Eigen::Matrix<double, 2, 3>& H_f, Eigen::Vector2d& r) {

	// Prepare all the required data.
	CamState* camState = m_map.getCamState(camID);
	FeatState* featState = m_map.getFeatState(featID);

	// Cam0 pose.
	Eigen::Matrix3d R_w_c0 = R(camState->m_q_C_G);
	Eigen::Vector3d& t_c0_w = camState->m_p_G;


	// 3d feature position in the world frame.
	// And its observation with the stereo cameras.
	const Eigen::Vector3d& p_w = featState->m_x_G;
	const Eigen::Vector2d& z = m_map.getPoint(camID, featID);

	// Convert the feature position from the world frame to
	// the cam0 and cam1 frame.
	Eigen::Vector3d p_c0 = R_w_c0 * (p_w - t_c0_w);

	// Compute the Jacobians.
	Eigen::Matrix<double, 2, 3> dz_dpc0 = Eigen::Matrix<double, 2, 3>::Zero();
	dz_dpc0(0, 0) = 1 / p_c0(2);
	dz_dpc0(1, 1) = 1 / p_c0(2);
	dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2)*p_c0(2));
	dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2)*p_c0(2));

	Eigen::Matrix<double, 3, 6> dpc0_dxc = Eigen::Matrix<double, 3, 6>::Zero();
	dpc0_dxc.leftCols(3) = Skew(p_c0);
	dpc0_dxc.rightCols(3) = -R_w_c0;

	Eigen::Matrix3d dpc0_dpg = R_w_c0;

	H_x = dz_dpc0*dpc0_dxc;
	H_f = dz_dpc0*dpc0_dpg;


	// Compute the residual.
	r = z - Eigen::Vector2d(p_c0(0) / p_c0(2), p_c0(1) / p_c0(2));

	return;
}


void Engine::measurementUpdate(
	const Eigen::MatrixXd& H, const Eigen::VectorXd& r) {

	if (H.rows() == 0 || r.rows() == 0) return;

	// Decompose the final Jacobian matrix to reduce computational
	// complexity as in Equation (28), (29).
	Eigen::MatrixXd H_thin;
	Eigen::VectorXd r_thin;

	if (H.rows() > H.cols()) {
		// Convert H to a sparse matrix.
		//Eigen::SparseMatrix<double> H_sparse = H.sparseView();

		//// Perform QR decompostion on H_sparse.
		//Eigen::SPQR<Eigen::SparseMatrix<double> > spqr_helper;
		//spqr_helper.setSPQROrdering(1/*SPQR_ORDERING_NATURAL*/);
		//spqr_helper.compute(H_sparse);

		//Eigen::MatrixXd H_temp;
		//Eigen::VectorXd r_temp;
		//(spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
		//(spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

		Eigen::SparseMatrix<double> H_sparse = H.sparseView();


		Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::NaturalOrdering<int>> spqr_helper;
		spqr_helper.compute(H_sparse);

		Eigen::MatrixXd H_temp;
		Eigen::VectorXd r_temp;
		(spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
		(spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

		H_thin = H_temp.topRows(15 + mv_camWindow.size() * 6);
		r_thin = r_temp.head(15 + mv_camWindow.size() * 6);

		//HouseholderQR<MatrixXd> qr_helper(H);
		//MatrixXd Q = qr_helper.householderQ();
		//MatrixXd Q1 = Q.leftCols(21+state_server.cam_states.size()*6);

		//H_thin = Q1.transpose() * H;
		//r_thin = Q1.transpose() * r;
	}
	else {
		H_thin = H;
		r_thin = r;
	}

	// Compute the Kalman gain.
	const Eigen::MatrixXd& P = m_P;
	Eigen::MatrixXd S = H_thin*P*H_thin.transpose() +
		OBSERVATION_NOISE * Eigen::MatrixXd::Identity(
			H_thin.rows(), H_thin.rows());
	//MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
	Eigen::MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
	Eigen::MatrixXd K = K_transpose.transpose();

	// Compute the error of the state.
	Eigen::VectorXd delta_x = K * r_thin;

	// Update the IMU state.
	const Eigen::VectorXd& delta_x_imu = delta_x.head<15>();

	if (//delta_x_imu.segment<3>(0).norm() > 0.15 ||
		//delta_x_imu.segment<3>(3).norm() > 0.15 ||
		delta_x_imu.segment<3>(6).norm() > 0.5 ||
		//delta_x_imu.segment<3>(9).norm() > 0.5 ||
		delta_x_imu.segment<3>(12).norm() > 1.0) {
		printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
		printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
		printf("Update change is too large.");
		//return;
	}

	const Eigen::Vector4d dq_imu =
		smallAngleQuaternion(delta_x_imu.head<3>());
	m_imuState.m_q_I_G = quaternionMultiplication(
		dq_imu, m_imuState.m_q_I_G);
	m_imuState.m_bg += delta_x_imu.segment<3>(3);
	m_imuState.m_v_G += delta_x_imu.segment<3>(6);
	m_imuState.m_ba += delta_x_imu.segment<3>(9);
	m_imuState.m_p_G += delta_x_imu.segment<3>(12);


	// Update the camera states.
	for (int i = 0; i < mv_camWindow.size(); ++i) {
		const Eigen::VectorXd& delta_x_cam = delta_x.segment<6>(15 + i * 6);
		const Eigen::Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
		CamState* camState = m_map.getCamState(mv_camWindow[i]);
		camState->m_q_C_G = quaternionMultiplication(
			dq_cam, camState->m_q_C_G);
		camState->m_p_G += delta_x_cam.tail<3>();
	}

	// Update state covariance.
	Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
	//m_P = I_KH*m_P*I_KH.transpose() +
	//  K*K.transpose()*Feature::observation_noise;
	m_P = I_KH*m_P;

	// Fix the covariance to be symmetric
	//Eigen::MatrixXd state_cov_fixed = (m_P +
	//	m_P.transpose()) / 2.0;
	//m_P = state_cov_fixed;

	return;
}

//easy method
void Engine::findRedundantCamStates(
	std::vector<CamIDType>& camIDList) {

	for (int i = 1; i < mv_camWindow.size() - 1;)
	{
		camIDList.push_back(mv_camWindow[i]);
		i += 3;
	}

	return;
}



void Engine::pruneCamStateBuffer() {

	if (mv_camWindow.size() < MAX_CAM_STATE)
		return;

	// Find two camera states to be removed.
	std::vector<CamIDType> rm_cam_state_ids(0);
	findRedundantCamStates(rm_cam_state_ids);

	// Find the size of the Jacobian matrix.
	int jacobian_row_size = 0;
	for (auto& item : m_featManager.featTracks) {
		FeatState* feature = m_map.getFeatState(item);
		// Check how many camera states to be removed are associated
		// with this feature.
		std::vector<CamIDType> involved_cam_state_ids;
		m_map.getCamStateList(item, involved_cam_state_ids);

		if (involved_cam_state_ids.size() == 0) continue;
		if (involved_cam_state_ids.size() == 1) {
			m_map.deleteMapNode(item, involved_cam_state_ids[0]);
		}

		if (!feature->m_isInit) {
			// Check if the feature can be initialize.
			if (!m_featManager.checkMotion(item)) {
				// If the feature cannot be initialized, just remove
				// the observations associated with the camera states
				// to be removed.
				for (const auto& cam_id : involved_cam_state_ids)
					m_map.deleteMapNode(cam_id, item);
				continue;
			}
			else {
				if (!m_featManager.initializePosition(item)) {
					for (const auto& cam_id : involved_cam_state_ids)
						m_map.deleteMapNode(cam_id, item);
					continue;
				}
			}
		}

		jacobian_row_size += 2 * involved_cam_state_ids.size() - 3;
	}

	//cout << "jacobian row #: " << jacobian_row_size << endl;

	// Compute the Jacobian and residual.
	Eigen::MatrixXd H_x = Eigen::MatrixXd::Zero(jacobian_row_size,
		15 + 6 * mv_camWindow.size());
	Eigen::VectorXd r = Eigen::VectorXd::Zero(jacobian_row_size);
	int stack_cntr = 0;

	for (auto& item : m_featManager.featTracks) {
		FeatState* feature = m_map.getFeatState(item);
		// Check how many camera states to be removed are associated
		// with this feature.

		std::vector<CamIDType> involved_cam_state_ids;
		m_map.getCamStateList(item, involved_cam_state_ids);

		if (involved_cam_state_ids.size() == 0) continue;

		Eigen::MatrixXd H_xj;
		Eigen::VectorXd r_j;
		featureJacobian(item, involved_cam_state_ids, H_xj, r_j);

		if (gatingTest(H_xj, r_j, involved_cam_state_ids.size())) {
			H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
			r.segment(stack_cntr, r_j.rows()) = r_j;
			stack_cntr += H_xj.rows();
		}

		for (const auto& cam_id : involved_cam_state_ids)
			m_map.deleteMapNode(cam_id, item);
	}

	H_x.conservativeResize(stack_cntr, H_x.cols());
	r.conservativeResize(stack_cntr);

	// Perform measurement update.
	measurementUpdate(H_x, r);

	for (int i = 0; i < rm_cam_state_ids.size(); ++i) 
	{
		int cam_state_start = 15 + 6 * i;
		int cam_state_end = cam_state_start + 6;

		// Remove the corresponding rows and columns in the state
		// covariance matrix.
		if (cam_state_end < m_P.rows()) {
			m_P.block(cam_state_start, 0, m_P.rows() - cam_state_end, m_P.cols()) = 
				m_P.block(cam_state_end, 0, m_P.rows() - cam_state_end, m_P.cols());

			m_P.block(0, cam_state_start,
				m_P.rows(),
				m_P.cols() - cam_state_end) =
				m_P.block(0, cam_state_end,
					m_P.rows(),
					m_P.cols() - cam_state_end);

			m_P.conservativeResize(
				m_P.rows() - 6, m_P.cols() - 6);
		}
		else {
			m_P.conservativeResize(
				m_P.rows() - 6, m_P.cols() - 6);
		}

		// Remove this camera state in the state vector.
		mv_camWindow.erase(mv_camWindow.begin() + 1 + 2 * i);
	}

	return;
}
