# System libs
import glob
import sys
import argparse

# Numeric libs
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from numpy import linalg as LA

import shutil
import os


# Computer Vision libs
from libs._3d.projection import *
from libs.detection.detection import *
from libs.siamMask.tools.siamMask import siamMask as maskTrack

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker
from std_msgs.msg       import UInt64

from cv_bridge import CvBridge, CvBridgeError


trf = T.Compose([T.ToPILImage(),
				 T.ToTensor()])


def prepareImage(img):
	
	#Make it square [we need this by now, we will remove it later when possible]
	center_y = int(img.shape[1]/2)
	h_half = int(img.shape[0]/2)
	return  img[:,center_y-h_half:center_y+h_half], center_y-h_half

class object:
	def __init__(self):		
		# 3D values
		self.centroid = None
		self.w = None
		self.h = None

class Tracker:
	def __init__(self, args):

		# Create object class
		self.glass = object()

		self.args = args

		self.cvBridge = CvBridge()
		self.c1 = projection(camId=1)
		self.c2 = projection(camId=2)

		self.track1 = None
		self.track2 = None
		self.fr1 = 0
		self.fr2 = 0
		self.fr = 0
		self.cam1_state = "wait"
		self.cam2_state = "wait"

		self.cam1 = dict.fromkeys(['rgb'])
		self.cam2 = dict.fromkeys(['rgb'])

		self.cam1_topic = "/camera1/color/image_raw"
		self.cam2_topic = "/camera2/color/image_raw"

		self.pub1_topic = "/siammask1/image"
		self.pub2_topic = "/siammask2/image"

		self.ROI1 = self.seg1 = self.points1 = None
		self.ROI2 = self.seg2 = self.points2 = None

		self.offset = None

		self.initial_location = None

		# Load object detection model
		self.detectionModel = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=3)
		self.detectionModel.load_state_dict(torch.load('./data/models/coco_maskrcnn_resnet50_fpn_2cat.pth', map_location='cpu'))
		self.detectionModel.eval()
		self.detectionModel.cuda()

		# Remove conten from out/record directory
		if self.args.record:
			if os.path.exists('./data/record'):
				shutil.rmtree('./data/record')
			if not os.path.exists('./data/record'):
				os.makedirs('./data/record')


	def cam1_callback(self, data):
		if self.cam1_state == "ready" and self.cam2_state == "wait":
			return
		
		self.cam1['rgb'] = self.cvBridge.imgmsg_to_cv2(data, "bgr8")
				

		if self.fr1 == 0:
			try:
				
				# Calibration
				self.c1.cameraPose(self.cam1)

				# Detection
				rospy.loginfo('Detecting object from camera 1...')
				img_square, self.offset = prepareImage(self.cam1['rgb'])
				output = self.detectionModel([trf(img_square).cuda()])

				self.ROI1, self.seg1, self.points1, self.img2plotc1 = postProcessingDetection(self.c1.camId, img_square, output[0], draw=False)
				
				if self.ROI1 is not None:
					# Move all to full image. To remove when bug about make it square is solved
					self.ROI1[0,0] += self.offset
					self.ROI1[0,2] += self.offset
					self.points1[:,0] += self.offset

					# Tracking
					rospy.loginfo('Initialising tracking from camera 1...')
					self.track1 = maskTrack(self.ROI1)
					self.track1.run(self.cam1['rgb'], 'init')

					self.cam1_state = "ready"
					rospy.loginfo('Camera 1 ready!')
				else:
					return
			except:
				return

		self.fr1 += 1

	def cam2_callback(self, data):
		if self.cam2_state == "ready" and self.cam1_state == "wait":
			return
		
		self.cam2['rgb'] = self.cvBridge.imgmsg_to_cv2(data, "bgr8")

		if self.fr2 == 0:
			try:
				
				# Calibration
				self.c2.cameraPose(self.cam2)

				# Detection
				rospy.loginfo('Detecting object from camera 2...')


				img_square, self.offset = prepareImage(self.cam2['rgb'])
				rospy.sleep(0.5)	# Prevents both topics to call GPU simultaneously
				output = self.detectionModel([trf(img_square).cuda()])
				self.ROI2, self.seg2, self.points2, self.img2plotc2 = postProcessingDetection(self.c2.camId, img_square, output[0], draw=False)

				if self.ROI2 is not None:
					# Move all to full image. To remove when bug about make it square is solved
					self.ROI2[0,0] += self.offset
					self.ROI2[0,2] += self.offset
					self.points2[:,0] += self.offset

					# Tracking
					rospy.loginfo('Initialising tracking from camera 2...')

					self.track2 = maskTrack(self.ROI2)
					self.track2.run(self.cam2['rgb'], 'init')

					self.cam2_state = "ready"
					rospy.loginfo('Camera 2 ready!')
				else:
					return
			except:
				return
				
		self.fr2 += 1
	

	def init_ros_topics(self):

		# ROS topic: retrieve frames
		rospy.Subscriber(self.cam1_topic, Image, self.cam1_callback)
		rospy.Subscriber(self.cam2_topic, Image, self.cam2_callback)

		# ROS topic: to publish image result
		self.pub1 = rospy.Publisher(self.pub1_topic, Image, queue_size=10)
		self.pub2 = rospy.Publisher(self.pub2_topic, Image, queue_size=10)


		# ROS topic: to publish 3D object location estimation and its dimensions
		self.markerPub = rospy.Publisher('estimatedObject', Marker, queue_size=10)
		self.state = Marker()
		self.state.type = Marker.CUBE
		self.state.header.frame_id = "/base_link"
		self.state.color.a = 1.
		self.state.color.r = 0.0;
		self.state.color.g = 1.0;
		self.state.color.b = 0.0;

		# Time for evaluation
		self.initial_locationPub = rospy.Publisher('initialLocation', Marker, queue_size=10)
		self.initial_state = Marker()
		self.initial_state.type = Marker.CUBE
		self.initial_state.header.frame_id = "/base_link"
		self.initial_state.color.a = 1.
		self.initial_state.color.r = 0.0;
		self.initial_state.color.g = 1.0;
		self.initial_state.color.b = 0.0;


	def getObjectDimensions(self):
		centroid1 = getCentroid(self.c1, self.seg1)
		centroid1[0] += self.offset
		centroid2 = getCentroid(self.c2, self.seg2)
		centroid2[0] += self.offset

		centroid = cv2.triangulatePoints(self.c1.extrinsic['rgb']['projMatrix'], self.c2.extrinsic['rgb']['projMatrix'], centroid1, centroid2).transpose()
		centroid /= centroid[:,-1].reshape(-1,1)
		centroid = centroid[:,:-1].reshape(-1)

		# Save initial location
		self.initial_location = np.matmul(self.camera_robot_transformation, np.append(copy.deepcopy(centroid).squeeze(), 1.).reshape(4,1))
		self.initial_state.pose.position.x = self.initial_location[0]
		self.initial_state.pose.position.y = self.initial_location[1]
		self.initial_state.pose.position.z = self.initial_location[2]
		self.initial_state.scale.x = 0
		self.initial_state.scale.y = 0
		self.initial_state.scale.z = 0
		
		wb_c1, h_c1 = getObjectDimensions(self.c1, self.seg1, self.cam1['rgb'], centroid, self.offset, 0, draw=False)
		wb_c2, h_c2 = getObjectDimensions(self.c2, self.seg2, self.cam2['rgb'], centroid, self.offset, 0, draw=False)
		wm_c1, _ = getObjectDimensions(self.c1, self.seg1, self.cam1['rgb'], centroid, self.offset, centroid[2], draw=False)
		wm_c2, _ = getObjectDimensions(self.c2, self.seg2, self.cam2['rgb'], centroid, self.offset, centroid[2], draw=False)
		
		wb = (wb_c1+wb_c2)/2
		wm = (wm_c1+wm_c2)/2
		wt = LA.norm(wb - wm) + wm
		h = (h_c1+h_c2)/2

		print('\nEstimated object dimenions: w_t={:.0f}mm, w_b={:.0f}mm, h={:.0f}mm\n'.format(wt*1000, wb*1000, h*1000))
		self.glass.w = wm
		self.glass.h = h

	def run(self):

		# Initialise subscribing topics
		self.init_ros_topics()
		
		# Read file to calibrate camera to robot
		f = open('./data/calibration/cameras_robot.pckl', 'rb')
		self.camera_robot_transformation = pickle.load(f)
		f.close()

		# Wait for cameras to be ready before going ahead
		while ((self.cam1_state != "ready") or (self.cam2_state != "ready")):
			continue

		# Estimate object dimensions [only on the first frame]
		rospy.loginfo('Estimating object dimensions...')
		self.getObjectDimensions()

		rate = rospy.Rate(30)
		while not rospy.is_shutdown():

			# Track in 2D
			img1 = self.track1.run(self.cam1['rgb'], 'track')
			img2 = self.track2.run(self.cam2['rgb'], 'track')

			# Triangulate to get 3D centroid
			self.glass, img1, img2 = get3D(self.c1, self.c2, self.track1.mask, self.track2.mask, self.glass, img1, img2, drawCentroid=True, drawDimensions=False)

			# ROS publish results
			self.pub1.publish(self.cvBridge.cv2_to_imgmsg(img1, encoding="passthrough"))
			self.pub2.publish(self.cvBridge.cv2_to_imgmsg(img2, encoding="passthrough"))

			########### 
			# PUBLISH #
			###########
			# Change refence system from cameras to robot
			self.glass.centroid = np.matmul(self.camera_robot_transformation, np.append(self.glass.centroid, 1.).reshape(4,1))

			self.initial_locationPub.publish(self.initial_state)

			# 3D marker message
			self.state.pose.position.x = self.glass.centroid[0,0]
			self.state.pose.position.y = self.glass.centroid[1,0]
			self.state.pose.position.z = self.glass.centroid[2,0]
			self.state.scale.x = self.glass.w
			self.state.scale.y = self.glass.w
			self.state.scale.z = self.glass.h
			self.markerPub.publish(self.state)

			if self.args.record:
				cv2.imwrite('./data/record/c1_track_{}.png'.format(self.fr1), img1)
				cv2.imwrite('./data/record/c2_track_{}.png'.format(self.fr2), img2)
				
			rate.sleep()

		rospy.spin() # Keep python from exiting until this node is stopped
		

if __name__ == '__main__':
	print('Initialising:')
	print('Python {}.{}'.format(sys.version_info[0], sys.version_info[1]))
	print('OpenCV {}'.format(cv2.__version__))
	print('PyTorch {}'.format(torch.__version__))
	print('Torchvision {}'.format(torchvision.__version__))

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--record', type=int, choices=[0,1], default=0)
	args = parser.parse_args()

	rospy.init_node('baselineCV', anonymous=True)

	track = Tracker(args)
	track.run()
