#pragma once
//reference from orbslam viewer

#include <vector>
#include <thread>
#include <pangolin/pangolin.h>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

extern std::vector<Eigen::Vector3d> poses;
extern std::vector<Eigen::Vector3d> poses1;
extern std::mutex pos_mut;

void run();