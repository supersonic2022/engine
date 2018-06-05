#include "visualization.h"
#include <iostream>

using namespace std;

std::vector<Eigen::Vector3d> poses;
std::vector<Eigen::Vector3d> poses1;
std::vector<Eigen::Vector3d> gts;
std::mutex pos_mut;

const int width = 1024;
const int height = 768;
const double fps = 30;
const double mT = 1e3 / fps;

void getGT()
{
	string path = "W:/vio/datasets/MH_01_easy/mav0/state_groundtruth_estimate0/";
	ifstream gt_file(path + "data.csv");
	if (!gt_file.good())
		cerr << " gt csv file not found !" << endl;

	string cur_line;
	getline(gt_file, cur_line); // first line delete
	while (getline(gt_file, cur_line, ','))
	{
		if (cur_line == "") break;	
		Eigen::Vector3d gt;
		for (int i = 0; i < 3; ++i)
		{
			getline(gt_file, cur_line, ',');
			gt[i] = stod(cur_line.c_str());
		}
		gts.push_back(gt);
		getline(gt_file, cur_line);
	}
}

void drawGT()
{
	if (!gts.size())
		return;

	//std::cout << "kf size = " << kfs.size() << std::endl;

	bool fisrt_gt = true;
	GLfloat ow1[3], ow2[3];

	int size = gts.size();

	for (int i = 0; i < size; ++i)
	{
		ow2[0] = gts[i][0];
		ow2[1] = gts[i][1];
		ow2[2] = gts[i][2];

		if (fisrt_gt)
			fisrt_gt = false;
		else
		{
			glLineWidth(1.0f);

			float r = (float)i / size;
			float g = 0.0f;
			float b = 0.0f;

			glColor3f(r, g, b);
			glBegin(GL_LINES);
			glVertex3f(ow1[0], ow1[1], ow1[2]);
			glVertex3f(ow2[0], ow2[1], ow2[2]);
			glEnd();
		}

		ow1[0] = ow2[0];
		ow1[1] = ow2[1];
		ow1[2] = ow2[2];
	}
}

void drawKFs()
{
	if (!poses.size())
		return;

	//std::cout << "kf size = " << kfs.size() << std::endl;

	bool fisrt_kf = true;
	GLfloat ow1[3], ow2[3];

	pos_mut.lock();
	int size = poses.size();
	pos_mut.unlock();

	for (int i = 0; i < size; ++i)
	{
		ow2[0] = poses[i][0];
		ow2[1] = poses[i][1];
		ow2[2] = poses[i][2];

		if (fisrt_kf)
			fisrt_kf = false;
		else
		{
			glLineWidth(1.0f);

			float r = (float)i / size;
			float g = 0.0f;
			float b = 0.0f;

			glColor3f(r, g, b);
			glBegin(GL_LINES);
			glVertex3f(ow1[0], ow1[1], ow1[2]);
			glVertex3f(ow2[0], ow2[1], ow2[2]);
			glEnd();
		}

		ow1[0] = ow2[0];
		ow1[1] = ow2[1];
		ow1[2] = ow2[2];
	}
}

void drawKFs1()
{
	if (!poses1.size())
		return;

	//std::cout << "kf size = " << kfs.size() << std::endl;

	bool fisrt_kf = true;
	GLfloat ow1[3], ow2[3];

	pos_mut.lock();
	int size = poses1.size();
	pos_mut.unlock();

	for (int i = 0; i < size; ++i)
	{
		ow2[0] = poses1[i][0];
		ow2[1] = poses1[i][1];
		ow2[2] = poses1[i][2];

		if (fisrt_kf)
			fisrt_kf = false;
		else
		{
			glLineWidth(1.0f);

			float r = 0.0f;
			float g = (float)i / size;
			float b = 1.0f;

			glColor3f(r, g, b);
			glBegin(GL_LINES);
			glVertex3f(ow1[0], ow1[1], ow1[2]);
			glVertex3f(ow2[0], ow2[1], ow2[2]);
			glEnd();
		}

		ow1[0] = ow2[0];
		ow1[1] = ow2[1];
		ow1[2] = ow2[2];
	}
}

void run()
{
	getGT();

	pangolin::CreateWindowAndBind("viwer", width, height);
	glEnable(GL_DEPTH_TEST);

	pangolin::OpenGlRenderState s_cam(
		pangolin::ProjectionMatrix(width, height, 500, 500, width / 2, height / 2, 0.1, 1000),
		pangolin::ModelViewLookAt(0, -0.7, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
	);

	pangolin::View& d_cam = pangolin::CreateDisplay()
		.SetBounds(0.0, 1.0, 0.0, 1.0, -(double)width / height)
		.SetHandler(new pangolin::Handler3D(s_cam));

	pangolin::OpenGlMatrix Twc;
	Twc.SetIdentity();

	while (1)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		d_cam.Activate(s_cam);
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

		drawGT();
		drawKFs();
		//drawKFs1();

		pangolin::FinishFrame();

		cv::waitKey(1);
	}

}

