# System libs
import glob
import sys
import os
import copy
import argparse

# Numeric libs
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from numpy import linalg as LA

# Computer Vision libs
from libs._3d.projection import *
from libs.detection.detection import *
from libs.siamMask.tools.siamMask import siamMask as maskTrack
from libs.graspAreas.grasp import RangeColorDetector, BackProjectionColorDetector


import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension

from cv_bridge import CvBridge, CvBridgeError


trf = T.Compose([T.ToPILImage(),
				 #T.Resize(256),
				 #T.CenterCrop(224),
				 T.ToTensor()])


def prepareImage(img):
	#Make it square [we need this by now, we will remove it later when possible]
	center_y = int(img.shape[1]/2)
	h_half = int(img.shape[0]/2)
	return  copy.deepcopy(img[:,center_y-h_half:center_y+h_half]), center_y-h_half

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
		self.cam1_state = "wait"
		self.cam2_state = "wait"

		self.got_dims = False

		self.cam1 = dict.fromkeys(['rgb','ir1','ir2'])
		self.cam2 = dict.fromkeys(['rgb','ir1','ir2'])

		self.cam1_topic = 		"/camera1/color/image_raw"
		self.cam1_ir1_topic = 	"/camera1/infra1/image_rect_raw"
		self.cam1_ir2_topic = 	"/camera1/infra2/image_rect_raw"


		self.cam2_topic = 		"/camera2/color/image_raw"
		self.cam2_ir1_topic = 	"/camera2/infra1/image_rect_raw"
		self.cam2_ir2_topic = 	"/camera2/infra2/image_rect_raw"


		self.pub1_topic = "/siammask1/image"
		self.pub2_topic = "/siammask2/image"

		self.ROI1 = self.seg1 = self.points1 = None
		self.ROI2 = self.seg2 = self.points2 = None

		self.offset = None


		# Load object detection model
		# os.environ['TORCH_HOME'] = os.getcwd()
		# self.detectionModel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
		path = './libs/detection/coco_maskrcnn_resnet50_fpn_2cats.pth'
		self.detectionModel = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=3)
		self.detectionModel.load_state_dict(torch.load(path, map_location='cpu'))
		self.detectionModel.eval()
		self.detectionModel.cuda()

		# Skin detection for grasping points
		min_range = np.array([0, 48, 70], dtype = "uint8") 		# Lower HSV boundary of skin color
		max_range = np.array([20, 150, 255], dtype = "uint8") 	# Upper HSV boundary of skin color
		self.skin = RangeColorDetector(min_range, max_range) 	# Define the detector object

	def cam1_callback(self, data):
		if self.cam1_state == "ready" and self.cam2_state == "wait":
			return
		self.cam1['rgb'] = self.cvBridge.imgmsg_to_cv2(data, "bgr8")

		if not self.got_dims:
			# Get infrared frames for object dimensions estimation
			#self.cam1['ir1'] = (self.cvBridge.imgmsg_to_cv2(rospy.wait_for_message(self.cam1_ir1_topic, Image), "bgr16") / 256).astype('uint8')
			#self.cam1['ir2'] = (self.cvBridge.imgmsg_to_cv2(rospy.wait_for_message(self.cam1_ir2_topic, Image), "bgr16") / 256).astype('uint8')
			self.cam1['ir1'] = self.cvBridge.imgmsg_to_cv2(rospy.wait_for_message(self.cam1_ir1_topic, Image), "bgr8")
			self.cam1['ir2'] = self.cvBridge.imgmsg_to_cv2(rospy.wait_for_message(self.cam1_ir2_topic, Image), "bgr8")

		#cv2.imwrite('c1_rgb.png', self.cam1['rgb'])
		#cv2.imwrite('c1_ir1.png', self.cam1['ir1'])
		#cv2.imwrite('c1_ir2.png', self.cam1['ir2'])

		if self.fr1 == 0:
			try:
				# Calibration
				self.c1.cameraPose(self.cam1)

				# Detection
				rospy.loginfo('Detecting object from camera 1...')
				img_square, self.offset = prepareImage(self.cam1['rgb'])
				output = self.detectionModel([trf(img_square).cuda()])

				self.ROI1, self.seg1, self.points1 = postProcessingDetection(self.c1.camId, img_square, output[0], draw=True)

				if self.ROI1 is not None:
					# Move all to full image. To remove when bug about make it square is solved
					self.ROI1[0,0] += self.offset
					self.ROI1[0,2] += self.offset
					self.points1[:,0] += self.offset

					# Get range of colors from the glass to filter out from grasping (skin) detection
					self.glass_color_detector_c1 = BackProjectionColorDetector()#Defining the color detector object
					self.glass_color_detector_c1.setTemplate(self.cam1['rgb'][self.ROI1[0,1]:self.ROI1[0,3],self.ROI1[0,0]:self.ROI1[0,2],:]) #Set the template

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

		if not self.got_dims:
			# Get infrared frames for object dimensions estimation
			#self.cam2['ir1'] = (self.cvBridge.imgmsg_to_cv2(rospy.wait_for_message(self.cam2_ir1_topic, Image), "bgr16") / 256).astype('uint8')
			#self.cam2['ir2'] = (self.cvBridge.imgmsg_to_cv2(rospy.wait_for_message(self.cam2_ir2_topic, Image), "bgr16") / 256).astype('uint8')
			self.cam2['ir1'] = self.cvBridge.imgmsg_to_cv2(rospy.wait_for_message(self.cam2_ir1_topic, Image), "bgr8")
			self.cam2['ir2'] = self.cvBridge.imgmsg_to_cv2(rospy.wait_for_message(self.cam2_ir2_topic, Image), "bgr8")

		#cv2.imwrite('c2_rgb.png', self.cam2['rgb'])
		#cv2.imwrite('c2_ir1.png', self.cam2['ir1'])
		#cv2.imwrite('c2_ir2.png', self.cam2['ir2'])

		if self.fr2 == 0:
			try:
				# Calibration
				self.c2.cameraPose(self.cam2)

				# Detection
				rospy.loginfo('Detecting object from camera 2...')

				img_square, self.offset = prepareImage(self.cam2['rgb'])
				rospy.sleep(0.5)	# Prevents both topics to call GPU simultaneously
				output = self.detectionModel([trf(img_square).cuda()])
				self.ROI2, self.seg2, self.points2 = postProcessingDetection(self.c2.camId, img_square, output[0], draw=True)

				if self.ROI2 is not None:
					# Move all to full image. To remove when bug about make it square is solved
					self.ROI2[0,0] += self.offset
					self.ROI2[0,2] += self.offset
					self.points2[:,0] += self.offset

					# Get range of colors from the glass to filter out from grasping (skin) detection
					self.glass_color_detector_c2 = BackProjectionColorDetector()#Defining the color detector object
					self.glass_color_detector_c2.setTemplate(self.cam2['rgb'][self.ROI2[0,1]:self.ROI2[0,3],self.ROI2[0,0]:self.ROI2[0,2],:]) #Set the template

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
		self.markerPub = rospy.Publisher('estimatedObject', PointStamped, queue_size=10)
		self.dimPub = rospy.Publisher('estimatedDimensions', Float64MultiArray, queue_size=10)

		self.state = PointStamped()
		self.state.header.frame_id = "/board"

		self.dimensions = Float64MultiArray()
		self.dimensions.layout = MultiArrayLayout()
		self.dimensions.layout.data_offset = 0
		self.dimensions.layout.dim = [MultiArrayDimension()]
		self.dimensions.layout.dim[0].size = 4
		self.dimensions.data = [0,0,0,0]

	def getObjectDimensions_stereo(self):
		img_c1_ir1, offset = prepareImage(self.cam1['ir1'])
		img_c1_ir2, _      = prepareImage(self.cam1['ir2'])
		img_c2_ir1, _      = prepareImage(self.cam2['ir1'])
		img_c2_ir2, _      = prepareImage(self.cam2['ir2'])

		points_c1_ir1 = None
		points_c1_ir2 = None
		points_c2_ir1 = None
		points_c2_ir2 = None

		output = None
		output = self.detectionModel([trf(img_c1_ir1).cuda(), trf(img_c1_ir2).cuda()])
		ROI_c1_ir1, seg_c1_ir1, points_c1_ir1 = postProcessingDetection(self.c1.camId, img_c1_ir1, output[0], ir=1, draw=True)
		ROI_c1_ir2, seg_c1_ir2, points_c1_ir2 = postProcessingDetection(self.c1.camId, img_c1_ir2, output[1], ir=2, draw=True)

		output = None	# This might be needed on small GPUs
		output = self.detectionModel([trf(img_c2_ir1).cuda(), trf(img_c2_ir2).cuda()])
		ROI_c2_ir1, seg_c2_ir1, points_c2_ir1 = postProcessingDetection(self.c2.camId, img_c2_ir1, output[0], ir=1, draw=True)
		ROI_c2_ir2, seg_c2_ir2, points_c2_ir2 = postProcessingDetection(self.c2.camId, img_c2_ir2, output[1], ir=2, draw=True)

		if points_c1_ir1 is None or points_c1_ir2 is None or points_c2_ir1 is None or points_c2_ir2 is None:
			return None, None

		points_c1_ir1[:,0] += offset
		points_c1_ir2[:,0] += offset
		points_c2_ir1[:,0] += offset
		points_c2_ir2[:,0] += offset

		ROI_c1_ir1[0,0] += offset
		ROI_c1_ir1[0,2] += offset
		ROI_c2_ir1[0,0] += offset
		ROI_c2_ir1[0,2] += offset

		#w_c1, h_c1 = self.c1.getObjectDimensions_stereo_v1(seg_c1_ir1, seg_c1_ir2, points_c1_ir1, points_c1_ir2, self.cam1, offset, draw=True)
		#w_c2, h_c2 = self.c2.getObjectDimensions_stereo_v1(seg_c2_ir1, seg_c2_ir2, points_c2_ir1, points_c2_ir2, self.cam2, offset, draw=True)
		w_c1, h_c1 = getObjectDimensions_stereo_v2(self.c1, seg_c1_ir1, seg_c1_ir2, points_c1_ir1, points_c1_ir2, self.cam1['ir1'], self.cam1['ir2'], offset, draw=True)
		w_c2, h_c2 = getObjectDimensions_stereo_v2(self.c2, seg_c2_ir1, seg_c2_ir2, points_c2_ir1, points_c2_ir2, self.cam2['ir1'], self.cam2['ir2'], offset, draw=True)

		# print('\nEstimated object dimenions: w={} x h={} cms\n'.format((w_c1+w_c2)/2*100, (h_c1+h_c2)/2*100))
		# self.glass.w = (w_c1+w_c2)/2
		# self.glass.h = (h_c1+h_c2)/2

		# self.dimensions.data = [self.glass.w, self.glass.h, 0, 0]
		# # self.dimensions.data = [0.095, 0.12, 0, 0]
		w = (w_c1+w_c2)/2
		h = (h_c1+h_c2)/2

		return w, h

	def getObjectDimensions(self):
		centroid1 = getCentroid(self.c1, self.seg1)
		centroid1[0] += self.offset
		centroid2 = getCentroid(self.c2, self.seg2)
		centroid2[0] += self.offset

		centroid = cv2.triangulatePoints(self.c1.extrinsic['rgb']['projMatrix'], self.c2.extrinsic['rgb']['projMatrix'], centroid1, centroid2).transpose()
		centroid /= centroid[:,-1].reshape(-1,1)
		centroid = centroid[:,:-1].reshape(-1)

		wb_c1, h_c1 = getObjectDimensions_v3(self.c1, self.seg1, self.cam1['rgb'], centroid, self.offset, 0, draw=False)
		wb_c2, h_c2 = getObjectDimensions_v3(self.c2, self.seg2, self.cam2['rgb'], centroid, self.offset, 0, draw=False)
		wm_c1, _ = getObjectDimensions_v3(self.c1, self.seg1, self.cam1['rgb'], centroid, self.offset, centroid[2], draw=False)
		wm_c2, _ = getObjectDimensions_v3(self.c2, self.seg2, self.cam2['rgb'], centroid, self.offset, centroid[2], draw=False)

		wb = (wb_c1+wb_c2)/2
		wm = (wm_c1+wm_c2)/2
		wt = LA.norm(wb - wm) + wm
		h = (h_c1+h_c2)/2

		return wb, wm, wt, h

	def testStereo(self):
		self.c1.drawGrid(self.cam1, 'rgb')
		self.c1.drawGrid(self.cam1, 'ir1')
		self.c1.drawGrid(self.cam1, 'ir2')

		self.c2.drawGrid(self.cam2, 'rgb')
		self.c2.drawGrid(self.cam2, 'ir1')
		self.c2.drawGrid(self.cam2, 'ir2')

		#self.c1.draw3DPoint(self.cam1, np.array((0.4,0.24,0.23)).reshape(1,3))
		#self.c2.draw3DPoint(self.cam2, np.array((0.4,0.24,0.23)).reshape(1,3))

		#self.c1.testRGB(self.cam1, self.cam2, self.c1, self.c2)
		#self.c1.testStereo(self.cam1, self.cam2, self.c2)
		#self.c2.testStereo(self.cam2, self.cam1, self.c1)
		#assert 1==0

	def run(self):

		# Initialise subscribing topics
		self.init_ros_topics()

		# # Read file to calibrate camera to robot
		# f = open('./data/calibration/cameras_robot.pckl', 'rb')
		# camera_robot_transformation = pickle.load(f)
		# f.close()

		# Wait for cameras to be ready before going ahead
		while ((self.cam1_state != "ready") or (self.cam2_state != "ready")):
			continue

		# self.testStereo() # Function for debuging

		cv2.imwrite('./data/out/c1_rgb.png', self.cam1['rgb'])
		cv2.imwrite('./data/out/c1_ir1.png', self.cam1['ir1'])
		cv2.imwrite('./data/out/c1_ir2.png', self.cam1['ir2'])

		cv2.imwrite('./data/out/c2_rgb.png', self.cam2['rgb'])
		cv2.imwrite('./data/out/c2_ir1.png', self.cam2['ir1'])
		cv2.imwrite('./data/out/c2_ir2.png', self.cam2['ir2'])

		# Estimate object dimensions [only on the first frame]

		#### NEW OBJECT DIMENSION ESTIMATIONS
		rospy.loginfo('Estimating object dimensions...')
		# total_w = []
		# total_h = []
		# max_tries = 20
		# N = 0
		# for _ in range(max_tries):
		# 	w, h = self.getObjectDimensions_stereo()	# This is more accurate but requires stereo lenses
		# 	if w is None or h is None:
		# 		continue
		# 	print(w*100,h*100)
		# 	total_w = total_w + [w]
		# 	total_h = total_h + [h]
		# 	N = N + 1
		# 	if w > 0.2:
		# 		break
		# 	if N >= 5:
		# 		break
		# 	rospy.sleep(1.)
		# if N == 0:
		# 	raise Exception('Could not find glass for object estimation')
		# self.glass.w = np.median(total_w)#total_w/float(N)
		# self.glass.h = np.median(total_h)#total_h/float(N)
		try:
			wb, wm, wt, h = self.getObjectDimensions()
		except:
			wb, wm, wt, h = 0.07, 0.075, 0.09, 0.13
		self.glass.w, self.glass.h = wm, h
		self.got_dims = True

		# print('\nEstimated object dimenions: w={} x h={} cms\n'.format(self.glass.w*100, self.glass.h*100))
		print('\nEstimated object dimenions: w_b={:.2f}, w_m={:.2f}, w_t={:.2f}, h={:.2f} cms\n'.format(wb*100, wm*100, wt*100, h*100))

		self.dimensions.data = [self.glass.w, self.glass.h, 0, 0]
		# self.dimensions.data = [0.095, 0.12, 0, 0]
		#self.glass = getObjectDimensions_oneLense(self.c1, self.c2, self.seg1, self.seg2, self.points1, self.points2, self.glass)

		glass_pos = [0, 0, 0]
		glass_height_start = 0
		glass_height_end = 0
		a_filter = 0.5

		rate = rospy.Rate(20)
		while not rospy.is_shutdown():
			# Track in 2D
			img1 = self.track1.run(self.cam1['rgb'], 'track')
			img2 = self.track2.run(self.cam2['rgb'], 'track')

			# Triangulate to get 3D centroid
			self.glass, img1, img2 = get3D(self.c1, self.c2, self.track1.mask, self.track2.mask, self.glass, img1, img2, drawCentroid=True, drawDimensions=False)

			# Grasping areas
			angle, graspHeight = self.skin.detectGraspAreas(self.c1.cropGlassFrom3D(self.glass, self.cam1['rgb']), self.track1.mask, self.glass_color_detector_c1,
															self.c2.cropGlassFrom3D(self.glass, self.cam2['rgb']), self.track2.mask, self.glass_color_detector_c2,
															self.glass, draw=False)

			if angle is not None:
				self.dimensions.data[2] = a_filter * glass_height_start + (1. - a_filter) * graspHeight[0]
				self.dimensions.data[3] = graspHeight[1] #a_filter * glass_height_end + (1. - a_filter) * graspHeight[1]

				glass_height_start, glass_height_end = self.dimensions.data[2], self.dimensions.data[3]
			self.dimPub.publish(self.dimensions)

			# TODO weight estimation

		   	# ROS publish results
			self.pub1.publish(self.cvBridge.cv2_to_imgmsg(img1, encoding="passthrough"))
			self.pub2.publish(self.cvBridge.cv2_to_imgmsg(img2, encoding="passthrough"))

			# Change refence system from cameras to robot
			# self.glass.centroid = np.matmul(camera_robot_transformation, np.append(self.glass.centroid, 1.).reshape(4,1))

			# 3D marker message
			self.state.point.x = a_filter * glass_pos[0] + (1. - a_filter) * self.glass.centroid[0,0]
			self.state.point.y = a_filter * glass_pos[1] + (1. - a_filter) * self.glass.centroid[0,1]
			self.state.point.z = a_filter * glass_pos[2] + (1. - a_filter) * self.glass.centroid[0,2]
			self.markerPub.publish(self.state)

			glass_pos[0] = self.state.point.x
			glass_pos[1] = self.state.point.y
			glass_pos[2] = self.state.point.z

			# print('angle: ', angle)
			# print('grasp: ', graspHeight)

			if self.args.record:
				cv2.imwrite('data/record/c1_{}.png'.format(self.fr1), self.cam1['rgb'])
				cv2.imwrite('data/record/c2_{}.png'.format(self.fr2), self.cam2['rgb'])
				cv2.imwrite('data/record/c1_track_{}.png'.format(self.fr1), img1)
				cv2.imwrite('data/record/c2_track_{}.png'.format(self.fr2), img2)

			rate.sleep()

		# spin() simply keeps python from exiting until this node is stopped
		rospy.spin()

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

	rospy.init_node('siammask', anonymous=True)

	track = Tracker(args)
	track.run()