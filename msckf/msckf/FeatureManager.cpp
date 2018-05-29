#include "FeatureManager.h"

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
		validTracks.clear();
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
				if (m_map->getFeatState(featTracks[i])->m_numCam >= DELAY_OBSERVATION)
					validTracks.push_back(featTracks[i]);
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
}