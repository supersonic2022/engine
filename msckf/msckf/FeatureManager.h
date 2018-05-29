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

	void processImage(CamIDType _camID, ImgData _img);

private:
	bool isFirst;
	Map* m_map;
	cv::Mat m_latestImg;
	cv::Mat m_mask;

	const int FEAT_PER_FRAME;
	const int RADIUS;

	std::vector<cv::Point2d> featPoints;
	std::vector<FeatIDType> featTracks;
	std::vector<FeatIDType> lostTracks;
	std::vector<FeatIDType> invalidTracks;
};
