# Baseline in Setup 1

## Instructions

The computer vision algorithm is the same as in S2 (minor updates were made to publish the appropriate data to ROS: see baseline_cv.py). Please refer to the README and the code in the S1 folder. The code for controlling the robotic manipulator differs. We provide a library for generic whole-body control (in C++) that we used for computing the inverse dynamics (or kinematics) of our manipulators. The [whc](https://github.com/costashatz/whc) library depends on the [DART](http://dartsim.github.io/) simulator (and [robot_dart](https://github.com/resibots/robot_dart)) and is a generic library for whole-body inverse kinematics and/or dynamics. We also provide a general sketch of our control loop (based on ROS) so that people can re-implement our baseline on their own robot (see robot_control.cpp).


## Camera calibration and poses
The file _calibration.csv_ contains the calibration data and the camera poses for each sensor (RGB, Depth, and Stereo infrared) for both IntelReal Sense devices.

Format:

*Columns

 fx,fy,cx,cy,R11,R12,R13,R21,R22,R23,R31,R32,R33,T1,T2,T3.
 
*Rows indicate the sensor
  1. camera 1 rgb
  2. camera 1 infrared 1
  3. camera 1 infrared 2
  4. camera 2 rgb
  5. camera 2 infrared 1
  6. camera 2 infrared 2
