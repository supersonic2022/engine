#pragma once

#include <Eigen/Eigen>
//include to use eigen stl container
#include <Eigen/StdVector>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <map>
#include <queue>

#define LOGE printf
#define LOGI printf

/*
reference:
[1]Quaternion kinematics for the error-state Kalman filter
[2]A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation
[3]Indirect Kalman Filter for 3D Attitude Estimation
*/

typedef Eigen::Matrix < double, 15, 15 > Matrix15d;
typedef Eigen::Matrix < double, 12, 12 > Matrix12d;
typedef Eigen::Matrix < double, 15, 12 > Matrix15_12d;

const double OBSERVATION_NOISE = 0.01;
const int DELAY_OBSERVATION = 4;
//use for check motion
const int THRESH_TRANSLATION = 0.2;
//use for LM dump
const int INITIAL_DAMP = 1e-3;
//BA params
const int NUM_ITERATION = 10;
const double PRECISION = 5e-7;
const double HUBER_EPSILON = 0.01;


const int MAX_CAM_STATE = 30;

enum Type
{
	IMU = 0,
	IMG = 1
};
