### Tested on:
* Python 2.7
* OpenCV 4.1.0
* PyTorch 1.1.0
* Torchvision 0.3.0
* Intel Real Sense D435i
* UR5 robotic arm
* Robotiq 2F-85 2-finger gripper

### Setting up:
* Print a charuco board
* Modify the properties of the charuco board in : ./libs/_3d/projection.py [variable CHARUCO_BOARD]
* Download cup detection model and place in ./data/models/
~~~~
wget https://drive.google.com/open?id=1ImLTap_lYduAFz2Jzdebwdz71kpXT2ol
mv coco_maskrcnn_resnet50_fpn_2cat <projectDirectory>/data/models
~~~~ 

* Launch roscore. Open a new terminal, load your anaconda environment and run:
~~~~
source activate <environmentName>
roscore	
~~~~ 

* Launch rviz. Open a new terminal, load your anaconda environment and run:
~~~~
source activate <environmentName>
rosrun rviz rviz 
~~~~ 

* Launch the cameras [make sure of modifying the SN in the launching files]. Open a new terminal, load your anaconda environment and run:
~~~~
source activate <environmentName>
roslaunch launch/rs_cam1.launch 
roslaunch launch/rs_cam2.launch
~~~~ 


### Running Vision node:

* Estimate camera-robot calibration trough vision [this might not be needed, for example, if you are using motion capture system to calibrate cameras and robot]
~~~~
source activate <environmentName>
python estimate_camera_robot_transformation.py
~~~~

* Launch vision node. Open a new terminal, run all the commands in init_SiamMask.txt, load your anaconda environment and run:
~~~~
source activate <environmentName>
python baseline_cv.py
~~~~ 


### Running robotic node:

* Launch robot driver, gripper driver and configuration files. Open a new terminal, load your anaconda environment and run:
~~~~
roslaunch ur_modern_driver ur5_bringup.launch limited:=true robot_ip:=<ROBOTIP> [reverse_port:=REVERSE_PORT]
roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch limited:=true
sudo usermod -a -G dialout <yourUserName>
rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0
~~~~

* Launch robot baseline [be careful, robot might move move!]. Open a new terminal, load your anaconda environment and run:
~~~~
python baseline_robot.py
~~~~


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
