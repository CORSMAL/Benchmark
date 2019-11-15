import cv2
import numpy.ma as ma
import cv2.aruco as aruco

import numpy as np
import pickle
from numpy import linalg as LA
from numpy.linalg import inv

import math

import copy
import rospy
from sensor_msgs.msg import CameraInfo
from realsense2_camera.msg import Extrinsics

class projection:
	def __init__(self, camId):
		
		self.camId = camId

		self.intrinsic = dict.fromkeys(['rgb'])
		self.extrinsic = dict.fromkeys(['rgb'])
		self.distCoeffs = None

		self.extrinsic['rgb'] = dict.fromkeys(['rvec','tvec','projMatrix'])
			

	def getIntrinsicFromROS(self, data):

		if 'color' in data.header.frame_id:
			self.intrinsic['rgb'] = np.array(data.K).reshape(3,3)

		self.distCoeffs = np.zeros((1,5), dtype=np.float64)
		

	def getIntrinsicParameters(self):
		self.getIntrinsicFromROS(rospy.wait_for_message('/camera{}/color/camera_info'.format(self.camId), CameraInfo))	#wait_for_message read only once the topic
		

	def calibrateWithBoard(self, imgs, sensor, draw=False):

		# Constant parameters used in Aruco methods
		ARUCO_PARAMETERS = aruco.DetectorParameters_create()
		ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)

		# Create grid board object we're using in our stream
		CHARUCO_BOARD = aruco.CharucoBoard_create(
			squaresX=10, 
			squaresY=6, 
			squareLength=0.04, #in meters
			markerLength=0.03, #in meters
			dictionary=ARUCO_DICT)

		# grayscale image
		gray = cv2.cvtColor(imgs[sensor], cv2.COLOR_BGR2GRAY)

		# Detect Aruco markers
		corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

		# Refine detected markers
		# Eliminates markers not part of our board, adds missing markers to the board
		corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(image = gray,
																					board = CHARUCO_BOARD,
																					detectedCorners = corners,
																					detectedIds = ids,
																					rejectedCorners = rejectedImgPoints,
																					cameraMatrix = self.intrinsic[sensor],
																					distCoeffs = self.distCoeffs)

		#print('Found {} corners in C{} sensor {}'.format(len(corners), self.camId, sensor))
		imgs[sensor] = aruco.drawDetectedMarkers(imgs[sensor], corners, ids=ids, borderColor=(0, 0, 255))

		# Only try to find CharucoBoard if we found markers
		if ids is not None and len(ids) > 10:

			# Get charuco corners and ids from detected aruco markers
			response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners=corners,
																					markerIds=ids,
																					image=gray,
																					board=CHARUCO_BOARD)

			# Require more than 20 squares
			if response is not None and response > 20:
				# Estimate the posture of the charuco board, which is a construction of 3D space based on the 2D video 
				pose, self.extrinsic[sensor]['rvec'], self.extrinsic[sensor]['tvec'] = aruco.estimatePoseCharucoBoard(charucoCorners=charuco_corners, 
																													charucoIds=charuco_ids, 
																													board=CHARUCO_BOARD, 
																													cameraMatrix=self.intrinsic[sensor], 
																													distCoeffs=self.distCoeffs)
				if draw:
					imgs[sensor] = aruco.drawAxis(imgs[sensor], self.intrinsic[sensor], self.distCoeffs, self.extrinsic[sensor]['rvec'], self.extrinsic[sensor]['tvec'], 2)
					cv2.imwrite('./data/out/calib_C{}_{}.png'.format(self.camId,sensor), imgs[sensor])
		else:
			print('Calibration board is not fully visible for C{} sensor: {}'.format(self.camId, sensor))
			assert 1==0
		
		self.extrinsic[sensor]['rvec'] = cv2.Rodrigues(self.extrinsic[sensor]['rvec'])[0]
		self.extrinsic[sensor]['projMatrix'] = np.matmul(self.intrinsic[sensor], np.concatenate((self.extrinsic[sensor]['rvec'],self.extrinsic[sensor]['tvec']), axis=1))

	def cameraPose(self, _imgs):

		rospy.loginfo('Calibrating camera {} ...'.format(self.camId))
		
		imgs = copy.deepcopy(_imgs)

		# Estimate extrinsic parameters (need a calibration board within the field of view of all cameras)
		self.getIntrinsicParameters()
		self.calibrateWithBoard(imgs, 'rgb', draw=True)


def getCentroid(c, mask):

	# Get the largest contour
	contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
	largest_contour = max(contour_sizes, key=lambda x: x[0])[1]

	# Get centroid of the largest contour
	M = cv2.moments(largest_contour)

	try:
		centroid = np.array((M['m10']/M['m00'], M['m01']/M['m00']))
		return centroid
	except:
		print('Centroid not found')
		return None

def triangulate(c1, c2, point1, point2, undistort=True):

	if (point1.dtype != 'float64'):
		point1 = point1.astype(np.float64)

	if (point2.dtype != 'float64'):
		point2 = point2.astype(np.float64)

	point3d = cv2.triangulatePoints(c1.extrinsic['rgb']['projMatrix'], c2.extrinsic['rgb']['projMatrix'], point1.reshape(2,1), point2.reshape(2,1)).transpose()
	for point in point3d:
		point /= point[-1]
	return point3d.reshape(-1)


def get3D(c1, c2, mask1, mask2, glass, _img1=None, _img2=None, drawCentroid=False, drawDimensions=False):
	
	img1 = copy.deepcopy(_img1)
	img2 = copy.deepcopy(_img2)

	centr1 = getCentroid(c1, mask1)
	centr2 = getCentroid(c2, mask2)
	if centr1 is not None and centr2 is not None:
		glass.centroid = triangulate(c1, c2, centr1, centr2)[:-1].reshape(-1,3)

		# Draw centroid
		if drawCentroid:
			
			# Draw 2D centroid of tracking mask
			#cv2.circle(img1, tuple(centr1.astype(int)), 10, (0,128,0), -1)
			#cv2.circle(img2, tuple(centr2.astype(int)), 10, (0,128,0), -1)

			# Draw 3D centroid projected to image
			point1, _ = cv2.projectPoints(glass.centroid, c1.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'], c1.intrinsic['rgb'], c1.distCoeffs)
			point2, _ = cv2.projectPoints(glass.centroid, c2.extrinsic['rgb']['rvec'], c2.extrinsic['rgb']['tvec'], c2.intrinsic['rgb'], c2.distCoeffs)

			point1 = point1.squeeze().astype(int)
			point2 = point2.squeeze().astype(int)
			
			cv2.circle(img1, tuple(point1), 6, (128,0,0), -1)
			cv2.circle(img2, tuple(point2), 6, (128,0,0), -1)

		# Draw height and width lines
		if drawDimensions:

			# Get top/bottom points
			top = copy.deepcopy(glass.centroid)
			bottom = copy.deepcopy(glass.centroid)
			top[0,2] += glass.h/2	
			bottom[0,2] -= glass.h/2
			topC1, _ 	= cv2.projectPoints(top, c1.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'], c1.intrinsic['rgb'], c1.distCoeffs)
			bottomC1, _ = cv2.projectPoints(bottom, c1.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'], c1.intrinsic['rgb'], c1.distCoeffs)
			topC2, _ 	= cv2.projectPoints(top, c2.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'], c2.intrinsic['rgb'], c2.distCoeffs)
			bottomC2, _ = cv2.projectPoints(bottom, c2.extrinsic['rgb']['rvec'], c2.extrinsic['rgb']['tvec'], c2.intrinsic['rgb'], c2.distCoeffs)
			topC1 = topC1.squeeze().astype(int)
			bottomC1 = bottomC1.squeeze().astype(int)
			topC2 = topC2.squeeze().astype(int)
			bottomC2 = bottomC2.squeeze().astype(int)

			# Get rigth/left points
			right = copy.deepcopy(glass.centroid)
			left = copy.deepcopy(glass.centroid)
			right[0,0] += glass.w/2
			left[0,0] -= glass.w/2
			rightC1, _ = cv2.projectPoints(right, c1.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'], c1.intrinsic['rgb'], c1.distCoeffs)
			leftC1, _ = cv2.projectPoints(left, c1.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'], c1.intrinsic['rgb'], c1.distCoeffs)
			rightC2, _ = cv2.projectPoints(right, c2.extrinsic['rgb']['rvec'], c2.extrinsic['rgb']['tvec'], c2.intrinsic['rgb'], c2.distCoeffs)
			leftC2, _ = cv2.projectPoints(left, c2.extrinsic['rgb']['rvec'], c2.extrinsic['rgb']['tvec'], c2.intrinsic['rgb'], c2.distCoeffs)
			rightC1 = rightC1.squeeze().astype(int)
			leftC1 = leftC1.squeeze().astype(int)
			rightC2 = rightC2.squeeze().astype(int)
			leftC2 = leftC2.squeeze().astype(int)

			cv2.line(img1, tuple(topC1), tuple(bottomC1), (128,0,0), 2)
			cv2.line(img1, tuple(rightC1), tuple(leftC1), (128,0,0), 2)
			cv2.line(img2, tuple(topC2), tuple(bottomC2), (128,0,0), 2)
			cv2.line(img2, tuple(rightC2), tuple(leftC2), (128,0,0), 2)


	return glass, img1, img2

def getObjectDimensions(cam, _seg, _img, centroid, offset, atHeight, draw=False):

		# Sample 3D circunferences in world coordinate system at z = centroid
		step = 0.001 #1mm
		minDiameter = 0.01 #1cm
		maxDiameter = 0.15 #20cm
		radiuses = np.linspace(maxDiameter/2, minDiameter/2, num=int((maxDiameter-minDiameter)/step))
		angularStep = 18#degrees
		angles = np.linspace(0., 359., num=int((359.)/angularStep))
	
		h = centroid[2]*2.
		for radius in radiuses:
			seg2plot = copy.deepcopy(_seg).squeeze()
			seg = copy.deepcopy(_seg).squeeze()
			img = copy.deepcopy(_img)
			
			p_3d = []
			for angle_d in angles:
				angle = math.radians(angle_d)
				p_3d.append(np.array((centroid[0]+(radius*math.cos(angle)), centroid[1]+(radius*math.sin(angle)), atHeight)).reshape(1,3))

			# Reproject to image
			p_2d, _ = cv2.projectPoints(np.array(p_3d), cam.extrinsic['rgb']['rvec'], cam.extrinsic['rgb']['tvec'], cam.intrinsic['rgb'], cam.distCoeffs)
			p_2d = p_2d.squeeze().astype(int)

			# Displace to segmentation
			p_2d[:,0] -= offset

			if draw:
				for p in p_2d:
					cv2.circle(img, (int(p[0]), int(p[1])), 2, (0,0,255), -1)

			if draw:
				for p in p_2d:
					cv2.circle(seg2plot, (int(p[0]), int(p[1])), 2, (0,0,255), -1)
			
			areIn = seg[p_2d[:,1], p_2d[:,0]]

			# Check if imaged points are in the segmentation
			if np.count_nonzero(areIn) == areIn.shape[0]:
				return radius*2, h
			
			if draw:
				cv2.imwrite('./data/out/measuring_C{}_rad{:.5f}.png'.format(cam.camId, radius), img)
