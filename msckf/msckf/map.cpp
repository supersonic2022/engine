#include "map.h"

CamIDType CamState::s_generator = 0ull;
FeatIDType FeatState::s_generator = 0ull;

Eigen::Vector3d ImuState::s_g_G = Eigen::Vector3d::Zero();
double ImuState::s_gyro_noise = 0.0f;
double ImuState::s_acc_noise = 0.0f;
double ImuState::s_gyro_bias_noise = 0.0f;
double ImuState::s_acc_bias_noise = 0.0f;


bool Map::addMapNode(CamIDType _camID, FeatIDType _featID, Eigen::Vector2d _point)
{
	if (mm_camServer.find(_camID) == mm_camServer.end())
	{
		LOGE("error: camID [%lld] not exist\n", _camID);
		return false;
	}
	if (mm_featServer.find(_featID) == mm_featServer.end())
	{
		LOGE("error: featID [%lld] not exist\n", _featID);
		return false;
	}
	MapNode* t_node = new MapNode(_camID, _featID, _point);

	CamState& t_camState = *(mm_camServer[_camID]);
	if (!t_camState.p_featNode)
		t_camState.p_featNode = t_node;
	else
	{
		MapNode* t_curNode = t_camState.p_featNode;
		while (t_curNode->p_nextFeat && t_curNode->p_nextFeat->m_featID < _featID)
			t_curNode = t_curNode->p_nextFeat;
		MapNode* t_nextNode = t_curNode->p_nextFeat;
		t_curNode->p_nextFeat = t_node;
		if (t_nextNode)
			t_node->p_nextFeat = t_nextNode;
	}
	t_camState.m_numFeat += 1;

	FeatState& t_featState = *(mm_featServer[_featID]);
	if (!t_featState.p_camNode)
		t_featState.p_camNode = t_node;
	else
	{
		MapNode* t_curNode = t_featState.p_camNode;
		while (t_curNode->p_nextCam && t_curNode->p_nextCam->m_camID < _camID)
			t_curNode = t_curNode->p_nextCam;
		MapNode* t_nextNode = t_curNode->p_nextCam;
		t_curNode->p_nextCam = t_node;
		if (t_nextNode)
			t_node->p_nextCam = t_nextNode;
	}
	t_featState.m_numCam += 1;

#ifdef MapMask
	mvv_mapMask[_camID][_featID] = 1;
#endif

	return true;
}


bool Map::deleteMapNode(CamIDType _camID, FeatIDType _featID)
{
	if (mm_camServer.find(_camID) == mm_camServer.end())
	{
		LOGE("error: camID [%lld] not exist\n", _camID);
		return false;
	}
	if (mm_featServer.find(_featID) == mm_featServer.end())
	{
		LOGE("error: featID [%lld] not exist\n", _featID);
		return false;
	}

	CamState& t_camState = *(mm_camServer[_camID]);
	if (!t_camState.p_featNode)
	{
		LOGE("error: MapNode with featID [%lld] and camID [%lld] not exist\n", _featID, _camID);
		return false;
	}
	else
	{
		MapNode* t_curNode = t_camState.p_featNode;
		if (t_curNode->m_featID == _featID)
		{
			t_camState.p_featNode = t_curNode->p_nextFeat;
		}
		else
		{
			while (t_curNode->p_nextFeat && t_curNode->p_nextFeat->m_featID < _featID)
				t_curNode = t_curNode->p_nextFeat;
			MapNode* t_nextNode = t_curNode->p_nextFeat;
			if(!t_nextNode || t_nextNode->m_featID != _featID)
			{
				LOGE("error: MapNode with featID [%lld] and camID [%lld] not exist\n", _featID, _camID);
				return false;
			}
			else
				t_curNode->p_nextFeat = t_nextNode->p_nextFeat;
		}
	}
	t_camState.m_numFeat -= 1;

	FeatState& t_featState = *(mm_featServer[_featID]);
	if (!t_featState.p_camNode)
	{
		LOGE("error: MapNode with featID [%lld] and camID [%lld] not exist\n", _featID, _camID);
		return false;
	}
	else
	{
		MapNode* t_curNode = t_featState.p_camNode;
		if (t_curNode->m_camID == _camID)
		{
			t_featState.p_camNode = t_curNode->p_nextCam;
			delete t_curNode;
		}
		else
		{
			while (t_curNode->p_nextCam && t_curNode->p_nextCam->m_camID < _camID)
				t_curNode = t_curNode->p_nextCam;
			MapNode* t_nextNode = t_curNode->p_nextCam;
			if (!t_nextNode || t_nextNode->m_camID != _camID)
			{
				LOGE("error: MapNode with featID [%lld] and camID [%lld] not exist\n", _featID, _camID);
				return false;
			}
			else
			{
				MapNode* t_node = t_curNode->p_nextCam;
				t_curNode->p_nextCam = t_nextNode->p_nextCam;
				delete t_node;
			}
		}
	}
	t_featState.m_numCam -= 1;

#ifdef MapMask
	mvv_mapMask[_camID][_featID] = 0;
#endif

	return true;
}

void Map::clearMap()
{
	//TODO clear the whole map
}

#ifdef MapMask
void Map::showMapMask()
{
	for (int c = 0; c < mvv_mapMask.size(); c++)
	{
		for (int f = 0; f < mvv_mapMask[c].size(); f++)
		{
			LOGI("%d ", mvv_mapMask[c][f]);
		}
		LOGI("\n");
	}
}
#endif

CamIDType Map::addCamState(Eigen::Vector4d _q_C_G, Eigen::Vector3d _p_G)
{
	CamState* camState = new CamState();
	camState->m_p_G = _p_G;
	camState->m_q_C_G = _q_C_G;
	mm_camServer.insert(std::pair<CamIDType, CamState*>(camState->m_camID, camState));
	return camState->m_camID;
}

void Map::addNewFeatState(CamIDType _camID, std::vector<cv::Point2f>& _pts, std::vector<FeatIDType>& _ptID, int startIdx)
{
	for (int i = startIdx; i < _pts.size(); i++)
	{
		FeatState* featState = new FeatState();
		mm_featServer.insert(std::pair<FeatIDType, FeatState*>(featState->m_featID, featState));
		addMapNode(_camID, featState->m_featID, Eigen::Vector2d(_pts[i].x, _pts[i].y));
		_ptID.push_back(featState->m_featID);
	}
}

void Map::addMatchFeatState(CamIDType _camID, std::vector<cv::Point2f>& _pts, std::vector<FeatIDType>& _ptID)
{
	for (int i = 0; i < _pts.size(); i++)
	{
		addMapNode(_camID, _ptID[i], Eigen::Vector2d(_pts[i].x, _pts[i].y));
	}
}


