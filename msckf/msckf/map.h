#pragma once

#include "global.h"

//#define MapMask

typedef long long IDType;
typedef IDType CamIDType;
typedef IDType FeatIDType;

struct MapNode
{
	CamIDType m_camID;
	FeatIDType m_featID;

	//2d point with respect to camera ID and feature ID
	Eigen::Vector2d m_point;
	//next feature with same camera
	MapNode* p_nextFeat;
	//next camera with same feature
	MapNode* p_nextCam;

	MapNode(CamIDType _camID, FeatIDType _featID, Eigen::Vector2d _point) :
		m_camID(_camID), m_featID(_featID), m_point(_point),
		p_nextCam(nullptr), p_nextFeat(nullptr) {}
};

struct CamState
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	IDType m_camID;

	//rotation from global frame to camera frame
	Eigen::Vector4d m_q_C_G;
	//poistion of camera frame in global frame
	Eigen::Vector3d m_p_G;
	//wether the camera is deleted
	bool m_isDeleted;

	//camera id generator
	static CamIDType s_generator;
	
	//pointer to the MapNode correspond to the first observed feature
	MapNode* p_featNode;
	//number of oberserved feature
	int m_numFeat;

	CamState() :
		m_q_C_G(Eigen::Vector4d(0, 0, 0, 1)),
		m_p_G(Eigen::Vector3d::Zero()),
		m_isDeleted(false),
		p_featNode(nullptr),
		m_numFeat(0)
	{
		m_camID = s_generator;
		s_generator++;
	}
};

struct FeatState
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	FeatIDType m_featID;
	//3D feature point in global frame
	Eigen::Vector3d m_x_G;
	//whether the feature is initialized
	bool m_isInit;
	//whether the feature is deleted
	bool m_isDelete;

	//feature ID generator
	static IDType s_generator;
	
	//pointer to the MapNode correspond to the first observed camera
	MapNode* p_camNode;
	//number of observed camera
	int m_numCam;

	FeatState() :
		m_x_G(Eigen::Vector3d::Zero()),
		m_isInit(false),
		m_isDelete(false),
		p_camNode(nullptr),
		m_numCam(0)
	{
		m_featID = s_generator;
		s_generator++;
	}
};

struct ImuState
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	//rotation from global frame to imu frame(body frame)
	Eigen::Vector4d m_q_I_G;
	//poistion of imu frame in global frame
	Eigen::Vector3d m_p_G;
	//velocity of imu frame in global frame
	Eigen::Vector3d m_v_G;

	//imu biases in global frame
	Eigen::Vector3d m_ba;
	Eigen::Vector3d m_bg;
	
	//gravity in global frame, should be fixed at first
	static Eigen::Vector3d s_g_G;

	//fixed noise parameters
	static double s_gyro_noise;
	static double s_acc_noise;
	static double s_gyro_bias_noise;
	static double s_acc_bias_noise;

	ImuState() :
		m_q_I_G(Eigen::Vector4d(0, 0, 0, 1)),
		m_bg(Eigen::Vector3d::Zero()),
		m_v_G(Eigen::Vector3d::Zero()),
		m_ba(Eigen::Vector3d::Zero()),
		m_p_G(Eigen::Vector3d::Zero())
	{}
};

//The whole Map is like a matrix linked with list structure
class Map
{
public:
	Map() {}
	~Map() { 
		clearMap(); 
	}

#ifdef MapMask
	void showMapMask();
#endif

	bool addMapNode(CamIDType _camID, FeatIDType _featID, Eigen::Vector2d _point);
	bool deleteMapNode(CamIDType _camID, FeatIDType _featID);
	void clearMap();

	CamIDType addCamState(Eigen::Vector4d _q_C_G, Eigen::Vector3d _p_G);
	void addNewFeatState(CamIDType _camID, std::vector<cv::Point2d>& _pts, std::vector<FeatIDType>& _ptID, int startIdx = 0);
	void addMatchFeatState(CamIDType _camID, std::vector<cv::Point2d>& _pts, std::vector<FeatIDType>& _ptID);

	FeatState* getFeatState(FeatIDType _featID)
	{
		return mm_featServer[_featID];
	}

	CamState* getCamState(CamIDType _camID)
	{
		return mm_camServer[_camID];
	}

	Eigen::Vector2d getPoint(CamIDType _camID, FeatIDType _featID)
	{
		CamState* camState = mm_camServer[_camID];
		MapNode* node = camState->p_featNode;
		while (node != nullptr)
		{
			if (node->m_featID == _featID)
				return node->m_point;
			node = node->p_nextFeat;
		}
		return Eigen::Vector2d::Zero();
	}

	void getCamStateList(FeatIDType _featId, std::vector<CamIDType>& camIDList)
	{
		camIDList.clear();
		FeatState* p_featState = mm_featServer[_featId];
		MapNode* node = p_featState->p_camNode;
		do
		{
			camIDList.push_back(node->m_camID);
			node = node->p_nextCam;
		}
		while (node!= nullptr);
	}


private:
	std::map<CamIDType, CamState*> mm_camServer;
	std::map<FeatIDType, FeatState*> mm_featServer;

#ifdef MapMask
	std::vector<std::vector<int>> mvv_mapMask;
#endif

};