#include "FeatureManager.h"
#include "utility.h"

void FeatureManager::processImage(CamIDType _camID, ImgData _img)
{
	cv::Mat curImg = cv::imread(_img._img_dir, 0);
	if (isFirst)
	{
		featPoints.clear();
		cv::goodFeaturesToTrack(curImg, featPoints, FEAT_PER_FRAME, 0.01, RADIUS, cv::noArray());
		m_map->addNewFeatState(_camID, featPoints, featTracks);
		isFirst = false;
	}
	else
	{
		lostTracks.clear();
		invalidTracks.clear();
		m_mask.create(curImg.size(), CV_8UC1);
		m_mask.setTo(1);

		std::vector<cv::Point2d> curPoints;
		std::vector<uchar> status;
		std::vector<double> err;
		cv::calcOpticalFlowPyrLK(m_latestImg, curImg, featPoints, curPoints, status, err);
		
		int cnt = 0;
		for (int i = 0; i < status.size(); i++)
		{
			if (status[i])
			{
				featPoints[cnt] = featPoints[i];
				curPoints[cnt] = curPoints[i];
				featTracks[cnt] = featTracks[i];
				cnt++;
			}
			else if (m_map->getFeatState(featTracks[i])->m_numCam >= DELAY_OBSERVATION)
				lostTracks.push_back(featTracks[i]);
			else
				invalidTracks.push_back(featTracks[i]);
			
		}

		featPoints.resize(cnt);
		curPoints.resize(cnt);
		featTracks.resize(cnt);

		status.clear();
		cv::findFundamentalMat(featPoints, curPoints, status);

		cnt = 0;
		for (int i = 0; i < status.size(); i++)
		{
			if (status[i])
			{
				featPoints[cnt] = curPoints[i];
				cv::circle(m_mask, featPoints[cnt], RADIUS, 0, -1);
				featTracks[cnt] = featTracks[i];
				cnt++;
			}
			else if (m_map->getFeatState(featTracks[i])->m_numCam >= DELAY_OBSERVATION)
				lostTracks.push_back(featTracks[i]);
			else
				invalidTracks.push_back(featTracks[i]);
		}
		featPoints.resize(cnt);
		featTracks.resize(cnt);


		m_map->addMatchFeatState(_camID, featPoints, featTracks);

		if (cnt < FEAT_PER_FRAME)
		{
			std::vector<cv::KeyPoint> t_fasts;
			cv::FAST(curImg, t_fasts, 20, true);
			std::sort(t_fasts.begin(), t_fasts.end(), [](const cv::KeyPoint& kp1, const cv::KeyPoint& kp2)
			{
				if (kp1.response > kp2.response) return true;
				else return false;
			});

			for (int i = 0, n = 0; i < t_fasts.size() && n < FEAT_PER_FRAME - cnt; i++)
			{
				if (m_mask.at<uchar>(t_fasts[i].pt))
				{
					featPoints.push_back(t_fasts[i].pt);
					cv::circle(m_mask, t_fasts[i].pt, RADIUS, 0, -1);
					n++;
				}
			}
			m_map->addNewFeatState(_camID, featPoints, featTracks, cnt);
		}
	}
	curImg.copyTo(m_latestImg);

	checkAndInit();
}

bool FeatureManager::checkMotion(FeatIDType _featID){
	std::vector<CamIDType> camIDList;
	m_map->getCamStateList(_featID, camIDList);
	int numCam = camIDList.size();

	Eigen::Isometry3d first_cam_pose;
	first_cam_pose.linear() = R(m_map->getCamState(camIDList[0])->m_q_C_G.transpose());
	first_cam_pose.translation() = m_map->getCamState(camIDList[0])->m_p_G;


	Eigen::Isometry3d last_cam_pose;
	last_cam_pose.linear() = R(m_map->getCamState(camIDList[numCam-1])->m_q_C_G.transpose());
	last_cam_pose.translation() = m_map->getCamState(camIDList[numCam - 1])->m_p_G;

	// Get the direction of the feature when it is first observed.
	// This direction is represented in the world frame.
	
	Eigen::Vector2d point = m_map->getPoint(camIDList[0], _featID);
	Eigen::Vector3d feature_direction(point[0], point[1], 1.0);
	feature_direction = feature_direction / feature_direction.norm();
	feature_direction = first_cam_pose.linear()*feature_direction;

	// Compute the translation between the first frame
	// and the last frame. We assume the first frame and
	// the last frame will provide the largest motion to
	// speed up the checking process.
	Eigen::Vector3d translation = last_cam_pose.translation() -
		first_cam_pose.translation();
	double parallel_translation =
		translation.transpose()*feature_direction;
	Eigen::Vector3d orthogonal_translation = translation -
		parallel_translation*feature_direction;

	if (orthogonal_translation.norm() > THRESH_TRANSLATION)
		return true;
	else return false;
}

void FeatureManager::generateInitialGuess(
	const Eigen::Isometry3d& T_c1_c2, const Eigen::Vector2d& z1,
	const Eigen::Vector2d& z2, Eigen::Vector3d& p) {
	// Construct a least square problem to solve the depth.
	Eigen::Vector3d m = T_c1_c2.linear() * Eigen::Vector3d(z1(0), z1(1), 1.0);

	Eigen::Vector2d A(0.0, 0.0);
	A(0) = m(0) - z2(0)*m(2);
	A(1) = m(1) - z2(1)*m(2);

	Eigen::Vector2d b(0.0, 0.0);
	b(0) = z2(0)*T_c1_c2.translation()(2) - T_c1_c2.translation()(0);
	b(1) = z2(1)*T_c1_c2.translation()(2) - T_c1_c2.translation()(1);

	// Solve for the depth.
	double depth = (A.transpose() * A).inverse() * A.transpose() * b;
	p(0) = z1(0) * depth;
	p(1) = z1(1) * depth;
	p(2) = depth;
	return;
}

bool FeatureManager::initializePosition(FeatIDType _featID) {
	// Organize camera poses and feature observations properly.
	std::vector<Eigen::Isometry3d,
		Eigen::aligned_allocator<Eigen::Isometry3d> > cam_poses;
	std::vector<Eigen::Vector2d,
		Eigen::aligned_allocator<Eigen::Vector2d> > measurements;

	std::vector<CamIDType> camIDList;
	m_map->getCamStateList(_featID, camIDList);

	for (auto& camID : camIDList) {
		// TODO: This should be handled properly. Normally, the
		//    required camera states should all be available in
		//    the input cam_states buffer.
		CamState* camState = m_map->getCamState(camID);

		// Add the measurement.
		measurements.push_back(m_map->getPoint(camID, _featID));

		// This camera pose will take a vector from this camera frame
		// to the world frame.
		Eigen::Isometry3d cam0_pose;
		cam0_pose.linear() = R(camState->m_q_C_G).transpose();
		cam0_pose.translation() = camState->m_p_G;

		cam_poses.push_back(cam0_pose);
	}

	// All camera poses should be modified such that it takes a
	// vector from the first camera frame in the buffer to this
	// camera frame.
	Eigen::Isometry3d T_c0_w = cam_poses[0];
	for (auto& pose : cam_poses)
		pose = pose.inverse() * T_c0_w;

	// Generate initial guess
	Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
	generateInitialGuess(cam_poses[cam_poses.size() - 1], measurements[0],
		measurements[measurements.size() - 1], initial_position);
	Eigen::Vector3d solution(
		initial_position(0) / initial_position(2),
		initial_position(1) / initial_position(2),
		1.0 / initial_position(2));

	// Apply Levenberg-Marquart method to solve for the 3d position.
	double lambda = INITIAL_DAMP;
	int inner_loop_cntr = 0;
	int outer_loop_cntr = 0;
	bool is_cost_reduced = false;
	double delta_norm = 0;

	// Compute the initial cost.
	double total_cost = 0.0;
	for (int i = 0; i < cam_poses.size(); ++i) {
		double this_cost = 0.0;
		cost(cam_poses[i], solution, measurements[i], this_cost);
		total_cost += this_cost;
	}

	// Outer loop.
	do {
		Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
		Eigen::Vector3d b = Eigen::Vector3d::Zero();

		for (int i = 0; i < cam_poses.size(); ++i) {
			Eigen::Matrix<double, 2, 3> J;
			Eigen::Vector2d r;
			double w;

			jacobian(cam_poses[i], solution, measurements[i], J, r, w);

			if (w == 1) {
				A += J.transpose() * J;
				b += J.transpose() * r;
			}
			else {
				double w_square = w * w;
				A += w_square * J.transpose() * J;
				b += w_square * J.transpose() * r;
			}
		}

		// Inner loop.
		// Solve for the delta that can reduce the total cost.
		do {
			Eigen::Matrix3d damper = lambda * Eigen::Matrix3d::Identity();
			Eigen::Vector3d delta = (A + damper).ldlt().solve(b);
			Eigen::Vector3d new_solution = solution - delta;
			delta_norm = delta.norm();

			double new_cost = 0.0;
			for (int i = 0; i < cam_poses.size(); ++i) {
				double this_cost = 0.0;
				cost(cam_poses[i], new_solution, measurements[i], this_cost);
				new_cost += this_cost;
			}

			if (new_cost < total_cost) {
				is_cost_reduced = true;
				solution = new_solution;
				total_cost = new_cost;
				lambda = lambda / 10 > 1e-10 ? lambda / 10 : 1e-10;
			}
			else {
				is_cost_reduced = false;
				lambda = lambda * 10 < 1e12 ? lambda * 10 : 1e12;
			}

		} while (inner_loop_cntr++ <
			NUM_ITERATION && !is_cost_reduced);

		inner_loop_cntr = 0;

	} while (outer_loop_cntr++ <
		NUM_ITERATION &&
		delta_norm > PRECISION);

	// Covert the feature position from inverse depth
	// representation to its 3d coordinate.
	Eigen::Vector3d final_position(solution(0) / solution(2),
		solution(1) / solution(2), 1.0 / solution(2));

	// Check if the solution is valid. Make sure the feature
	// is in front of every camera frame observing it.
	bool is_valid_solution = true;
	for (const auto& pose : cam_poses) {
		Eigen::Vector3d position =
			pose.linear()*final_position + pose.translation();
		if (position(2) <= 0) {
			is_valid_solution = false;
			break;
		}
	}

	FeatState* featState = m_map->getFeatState(_featID);
	// Convert the feature position to the world frame.
	featState->m_x_G = T_c0_w.linear()*final_position + T_c0_w.translation();

	if (is_valid_solution)
		featState->m_isInit = true;

	return is_valid_solution;
}

void FeatureManager::cost(const Eigen::Isometry3d& T_c0_ci,
	const Eigen::Vector3d& x, const Eigen::Vector2d& z,
	double& e) const {
	// Compute hi1, hi2, and hi3 as Equation (37).
	const double& alpha = x(0);
	const double& beta = x(1);
	const double& rho = x(2);

	Eigen::Vector3d h = T_c0_ci.linear()*
		Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
	double& h1 = h(0);
	double& h2 = h(1);
	double& h3 = h(2);

	// Predict the feature observation in ci frame.
	Eigen::Vector2d z_hat(h1 / h3, h2 / h3);

	// Compute the residual.
	e = (z_hat - z).squaredNorm();
	return;
}

void FeatureManager::jacobian(const Eigen::Isometry3d& T_c0_ci,
	const Eigen::Vector3d& x, const Eigen::Vector2d& z,
	Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
	double& w) const {

	// Compute hi1, hi2, and hi3 as Equation (37).
	const double& alpha = x(0);
	const double& beta = x(1);
	const double& rho = x(2);

	Eigen::Vector3d h = T_c0_ci.linear()*
		Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
	double& h1 = h(0);
	double& h2 = h(1);
	double& h3 = h(2);

	// Compute the Jacobian.
	Eigen::Matrix3d W;
	W.leftCols<2>() = T_c0_ci.linear().leftCols<2>();
	W.rightCols<1>() = T_c0_ci.translation();

	J.row(0) = 1 / h3*W.row(0) - h1 / (h3*h3)*W.row(2);
	J.row(1) = 1 / h3*W.row(1) - h2 / (h3*h3)*W.row(2);

	// Compute the residual.
	Eigen::Vector2d z_hat(h1 / h3, h2 / h3);
	r = z_hat - z;

	// Compute the weight based on the residual.
	double e = r.norm();
	if (e <= HUBER_EPSILON)
		w = 1.0;
	else
		w = HUBER_EPSILON / (2 * e);

	return;
}

void FeatureManager::checkAndInit()
{
	valideTracks.clear();
	for (int i = 0; i < lostTracks.size(); i++)
	{
		if (!checkMotion(lostTracks[i]))
		{
			invalidTracks.push_back(lostTracks[i]);
			continue;
		}
		if (!initializePosition(lostTracks[i]))
		{
			invalidTracks.push_back(lostTracks[i]);
			continue;
		}
		valideTracks.push_back(lostTracks[i]);
	}
}