#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi

from numpy import linalg as LA

from tf.transformations import quaternion_from_euler, euler_from_quaternion
import logging
import os
import time

from std_msgs.msg       import UInt64


from visualization_msgs.msg import Marker
import numpy as np

from moveit_commander.conversions import pose_to_list

from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg_gripper
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input as inputMsg_gripper


def all_close(goal, actual, tolerance):
  """
  Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
  @param: goal       A list of floats, a Pose or a PoseStamped
  @param: actual     A list of floats, a Pose or a PoseStamped
  @param: tolerance  A float
  @returns: bool
  """
  all_equal = True
  if type(goal) is list:
	for index in range(len(goal)):
	  if abs(actual[index] - goal[index]) > tolerance:
		return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
	return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
	return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True

class MoveRobot(object):
	def __init__(self):
		super(MoveRobot, self).__init__()

		## First initialize `moveit_commander`_ and a `rospy`_ node:
		moveit_commander.roscpp_initialize(sys.argv)
		rospy.init_node('MoveRobot', anonymous=True)

		## Instantiate a `RobotCommander`_ object. This object is the outer-level interface to  the robot:
		robot = moveit_commander.RobotCommander()

		## Instantiate a `PlanningSceneInterface`_ object.  This object is an interface to the world surrounding the robot:
		scene = moveit_commander.PlanningSceneInterface()

		## This interface can be used to plan and execute motions on the Panda:
		group_name = "manipulator"
		group = moveit_commander.MoveGroupCommander(group_name)

		## We create a `DisplayTrajectory`_ publisher which is used later to publish trajectories for RViz to visualize:
		display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

		# Misc variables
		self.box_name = ''
		self.robot = robot
		self.scene = scene
		self.group = group
		self.display_trajectory_publisher = display_trajectory_publisher


		self.topic = '/estimatedObject'
		self.position = None
		self.ready = False

		self.gripper_topic = 'Robotiq2FGripperRobotOutput'

		# Initial pose
		self.pose_initial = geometry_msgs.msg.Pose()
		#q = quaternion_from_euler(-3.1143, -0.004, 1.5545)
		self.pose_initial.orientation.x = -0.688592372522
		self.pose_initial.orientation.y = -0.724833773664
		self.pose_initial.orientation.z = -0.0213073239923
		self.pose_initial.orientation.w = 0.0015946800291
		self.pose_initial.position.x = -0.119410771433
		self.pose_initial.position.y = .25
		self.pose_initial.position.z =  0.112867401896


		# Time for evaluation
		self.timePubB = rospy.Publisher('timeB', UInt64, queue_size=10)
		self.timePubD = rospy.Publisher('timeD', UInt64, queue_size=10)
		self.savedTimeB = None
		self.savedTimeD = None


	def callback(self, data):

		self.ready = False
		self.glass = data

		#Fix orientation
		self.glass.pose.position.y -= .2
		self.glass.pose.orientation.x = -0.688592372522
		self.glass.pose.orientation.y = -0.724833773664
		self.glass.pose.orientation.z = -0.0213073239923
		self.glass.pose.orientation.w = 0.0015946800291
		self.ready = True

	def callbackInitial(self, data):

		self.pose_delivery = data

		#Fix orientation
		self.pose_delivery.pose.position.y -= .2
		self.pose_delivery.pose.position.z -= .03
		self.pose_delivery.pose.orientation.x = -0.688592372522
		self.pose_delivery.pose.orientation.y = -0.724833773664
		self.pose_delivery.pose.orientation.z = -0.0213073239923
		self.pose_delivery.pose.orientation.w = 0.0015946800291
		
 	def go_to_relative(self, _goal, x=0, y=0, z=0, thr=0.01):
 		goal = copy.deepcopy(_goal)
 		if x!=0:
 			goal.position.x += x
		if y!=0:
			goal.position.y += y
		if z!=0:
			goal.position.z += z

		distance = self.go_to_pose(goal)
		return distance <= thr		

	def go_to(self, goal, thr=0.01):
		distance = self.go_to_pose(goal)
		return distance <= thr		

	def go_to_pose(self, goal):

		pose_goal = geometry_msgs.msg.Pose()
		pose_goal.orientation.x = goal.orientation.x
		pose_goal.orientation.y = goal.orientation.y
		pose_goal.orientation.z = goal.orientation.z
		pose_goal.orientation.w = goal.orientation.w
		pose_goal.position.x = goal.position.x
		pose_goal.position.y = goal.position.y
		pose_goal.position.z = goal.position.z 

		## Now, we call the planner to compute the plan and execute it.
		self.group.set_pose_target(pose_goal)

		tt = self.group.plan()
		#print('Planning to {}... took {} points'.format(self.glass, len(tt.joint_trajectory.points)))
		
		if len(tt.joint_trajectory.points) < 20 and len(tt.joint_trajectory.points) > 0: #shorter solutions seems to be safe/good
			#pass
			# Perform the motion
			self.group.execute(tt, wait=True)
			
			# Calling `stop()` ensures that there is no residual movement
			#self.group.stop()
			
			# It is always good to clear your targets after planning with poses. Note: there is no equivalent function for clear_joint_value_targets()
			#self.group.clear_pose_targets()

		#else:
		#	print('Not going...')

		# Return distance between current pose and goal pose
		self.current_pose = self.group.get_current_pose()
		current = np.array((self.current_pose.pose.position.x, self.current_pose.pose.position.y, self.current_pose.pose.position.z))
		goal = np.array((goal.position.x, goal.position.y, goal.position.z))
		distance = LA.norm(current - goal)
		#print('Distance to goal: {:.5f}'.format(distance))
		return distance

	def initGripper(self):
		self.gripper_msg.rACT = 1
		self.gripper_msg.rGTO = 1
		self.gripper_msg.rATR = 0
		self.gripper_msg.rSP = 255
		self.gripper_msg.rPR = 0	#open gripper
		self.gripper_msg.rFR = 150
		self.gripper_msg.rPR = 128
		self.pub_gripper.publish(self.gripper_msg)
		#rospy.sleep(0.1)

	def closeGripper(self, distance):
		#0: open
		#255: closed
		distance *= 1000 # distance to mm from meters
		self.gripper_msg.rPR = max(0, min(255, int(-3*distance+255)))
		self.pub_gripper.publish(self.gripper_msg)
		#rospy.sleep(0.1)

	def openGripper(self):
		#0: open
		#255: closed
		self.gripper_msg.rPR = 0
		self.pub_gripper.publish(self.gripper_msg)
		#rospy.sleep(0.1)

	def run(self):
			
		rospy.Subscriber(self.topic, Marker, self.callback)
		rospy.Subscriber('initialLocation', Marker, self.callbackInitial)

		rate = rospy.Rate(200)
		self.group.allow_replanning(True)
		self.group.set_planning_time(1.0)
		#self.group.set_planner_id("SBLkConfigDefault")

		# Gripper
		self.pub_gripper = rospy.Publisher(self.gripper_topic, outputMsg_gripper.Robotiq2FGripper_robot_output)

		self.gripper_msg = outputMsg_gripper.Robotiq2FGripper_robot_output()
		self.initGripper()

		print('Move to initial pose')
		mode = 'init'
		while not rospy.is_shutdown():

			if mode == 'init':
				if self.go_to(self.pose_initial):
					mode = 'pick'
					raw_input("Initial position reached. Click to start...")

			if self.ready is not True:
				continue
			
			if mode == 'pick':
				self.current_pose = self.group.get_current_pose() # Get current location of eef
				if self.go_to(self.glass.pose):
					self.closeGripper(self.glass.scale.x -0.005)
					mode = 'delivery'

					time = rospy.get_rostime()
					self.savedTimeB = time.secs*10**3 + (time.nsecs/10**6)	# in ms
					print('Grabbing glass at {}ms'.format(self.savedTimeB))


			elif mode == 'delivery':
				if self.go_to(self.pose_delivery.pose):
					self.openGripper()
					time = rospy.get_rostime()
					self.savedTimeD = time.secs*10**3 + (time.nsecs/10**6)	# in ms
					print('Completed at {}ms'.format(self.savedTimeD))
					mode = 'completing'

			elif mode == 'completing':
				if self.go_to_relative(self.pose_delivery.pose, y=-0.2):
					mode = 'done'

			self.timePubB.publish(self.savedTimeB)
			self.timePubD.publish(self.savedTimeD)

			if mode == 'done':
				break
		
			rate.sleep()

		return
		# spin() simply keeps python from exiting until this node is stopped
		rospy.spin()
		

if __name__ == '__main__':
	robot = MoveRobot()

	robot.run()





