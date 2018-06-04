#pragma once

#include "map.h"
#include "imudata.h"

class FeatureManager
{
public:
	FeatureManager(Map* _map) :
		m_map(_map), isFirst(true),
		FEAT_PER_FRAME(200),
		RADIUS(20)
	{}

	int processImage(CamIDType _camID, ImgData _img);

	void removeInvalid();
	bool checkMotion(FeatIDType _featID);
	int initializePosition(FeatIDType _featID);

	//tracks used for update
	std::vector<FeatIDType> valideTracks;
	std::vector<FeatIDType> featTracks;

private:

	int checkAndInit();

	void generateInitialGuess(
		const Eigen::Isometry3d& T_c1_c2, const Eigen::Vector2d& z1,
		const Eigen::Vector2d& z2, Eigen::Vector3d& p);



	void cost(const Eigen::Isometry3d& T_c0_ci,
		const Eigen::Vector3d& x, const Eigen::Vector2d& z,
		double& e) const;

	void jacobian(const Eigen::Isometry3d& T_c0_ci,
		const Eigen::Vector3d& x, const Eigen::Vector2d& z,
		Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
		double& w) const;



	bool isFirst;
	Map* m_map;
	cv::Mat m_latestImg;
	cv::Mat m_mask;

	const int FEAT_PER_FRAME;
	const int RADIUS;

	std::vector<FeatIDType> invalidTracks;
	std::vector<cv::Point2f> featPoints;
	std::vector<FeatIDType> lostTracks;

};
