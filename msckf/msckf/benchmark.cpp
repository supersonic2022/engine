#include "benchmark.h"

BenchmarkNode::BenchmarkNode()
{
	dataset = new EuRoCData("E:/euroc/mav0");
}

BenchmarkNode::~BenchmarkNode()
{
	delete dataset;
}

void BenchmarkNode::runFromFolder()
{
	//serialize read
	int imu_idx = 0;
	int img_idx = 0;

	double img_time = stod(dataset->img_timestamps[0][img_idx]);
	double imu_time = stod(dataset->imu_timestamps[0][imu_idx].first);
	double next_imu_time = stod(dataset->imu_timestamps[0][imu_idx + 1].first);

	while (img_time <= imu_time)
	{
		img_idx += 1;
		img_time = stod(dataset->img_timestamps[0][img_idx]);
	}

	while (imu_idx != dataset->imu_timestamps[0].size() - 2 && img_idx != dataset->img_timestamps[0].size() - 1)
	{
		while(imu_time < img_time && next_imu_time <= img_time)
		{
			ImuData imudata(dataset->imu_timestamps[0][imu_idx].second, (next_imu_time - imu_time) / 1e9);
			mq_imuQueue.push(imudata);
			mq_maskQueue.push(IMU);

			std::cout << "imu : " << dataset->imu_timestamps[0][imu_idx].first << std::endl;

			imu_time = next_imu_time;
			imu_idx += 1;
			next_imu_time = stod(dataset->imu_timestamps[0][imu_idx + 1].first);
		}
		
		if (fabs(imu_time - img_time) < 1e-6)
		{
			std::stringstream ss;
			ss << dataset->cam_data_files[0] << dataset->img_timestamps[0][img_idx] << ".png";
			ImgData imgdata(ss.str().c_str(), stod(dataset->img_timestamps[0][img_idx]));
			mq_imgQueue.push(imgdata);
			mq_maskQueue.push(IMG);

			std::cout << "image : " << dataset->img_timestamps[0][img_idx] << std::endl;
			img_idx += 1;
			img_time = stod(dataset->img_timestamps[0][img_idx]);
			continue;
		}
		else
		{
			double a = img_time - imu_time;
			double b = next_imu_time - img_time;
			double factor = b / (a + b);

			ImuData imudata(dataset->imu_timestamps[0][imu_idx].second, a / 1e9);
			mq_imuQueue.push(imudata);
			mq_maskQueue.push(IMU);

			std::cout << "imu : " << dataset->imu_timestamps[0][imu_idx].first << std::endl;

			std::stringstream ss;
			ss << dataset->cam_data_files[0] << dataset->img_timestamps[0][img_idx] << ".png";
			ImgData imgdata(ss.str().c_str(), stod(dataset->img_timestamps[0][img_idx]));
			mq_imgQueue.push(imgdata);
			mq_maskQueue.push(IMG);

			std::cout << "image : " << dataset->img_timestamps[0][img_idx] << std::endl;
			img_idx += 1;
			img_time = stod(dataset->img_timestamps[0][img_idx]);

			Eigen::Matrix<double, 6, 1> t_g_a = factor * dataset->imu_timestamps[0][imu_idx].second +
												(1 - factor) * dataset->imu_timestamps[0][imu_idx + 1].second;
			imudata = ImuData(t_g_a, b / 1e9);
			mq_imuQueue.push(imudata);
			mq_maskQueue.push(IMU);

			std::cout << "imu : " << dataset->img_timestamps[0][img_idx-1] << std::endl;

			imu_time = next_imu_time;
			imu_idx += 1;
			next_imu_time = stod(dataset->imu_timestamps[0][imu_idx + 1].first);
			continue;
		}
	}
}



