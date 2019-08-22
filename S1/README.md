### Tested on:
* Python 2.7
* OpenCV 4.1.0
* PyTorch 1.1.0
* Torchvision 0.3.0
* Intel Real Sense D435i
* UR5 robotic arm

### Setting up:
* Print a charuco board
* Modify the properties of the charuco board in : ./libs/_3d/projection.py [variable CHARUCO_BOARD]
* Download cup detection model and place in ./data/models/
~~~~
wget xxxxxxxxxxxxx
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