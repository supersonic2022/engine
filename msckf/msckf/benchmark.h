#pragma once

#include "global.h"
#include "imudata.h"
#include "EuRoCReader.h"

//simple synchronize class
class BenchmarkNode
{
public:

	EuRoCData* dataset;
	queue<ImgData> mq_imgQueue;
	queue<ImuData> mq_imuQueue;
	//0 for image and 1 for imu
	queue<Type> mq_maskQueue;

public:
	BenchmarkNode();
	~BenchmarkNode();
	void runFromFolder();
};