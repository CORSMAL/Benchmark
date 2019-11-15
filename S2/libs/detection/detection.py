import numpy as np
import copy
import cv2


COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'wine_glass', 'cup']


def postProcessingDetection(camId, _img, output, ir=1, draw=False):
	
	img = copy.deepcopy(_img)

	# Draw glasses and similar (e.g. vase, cup)
	ROI = []
	glassFound = False
	for i in range(0,len(output['labels'])):
		if output['scores'][i] >= 0.5 and output['labels'][i] in [1,2]:
			glassFound = True

			seg = (output['masks'][i,:,:,:] >= 0.5).cpu().numpy()

			ROI.append(output['boxes'][i].cpu().detach().numpy())
			if draw:
				cv2.rectangle(img,(output['boxes'][i][0],output['boxes'][i][1]),(output['boxes'][i][2],output['boxes'][i][3]), (255,0,0),3)
				cv2.putText(img, COCO_INSTANCE_CATEGORY_NAMES[output['labels'][i]], (output['boxes'][i][2], output['boxes'][i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), lineType=cv2.LINE_AA) 
				cv2.putText(img, str('{:.2f}'.format(output['scores'][i])), (output['boxes'][i][2], output['boxes'][i][3]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), lineType=cv2.LINE_AA) 
			break
	
	seg = np.transpose(seg*255, (1,2,0))
	
	# Get extreme points and draw
	if glassFound:
		points = getExtremePoints(seg)
		if draw:
			for i, point in enumerate(points):
				cv2.circle(img, tuple(point), 10, (0,0,255), -1)
				cv2.putText(img, str(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

			img = 0.5* img + 0.5*seg
			#img[:, :, 1] = (seg.squeeze() > 0) * 255 + (seg.squeeze() == 0) * img[:, :, 1]

			cv2.imwrite('./data/out/seg_C{}_ir{}.png'.format(camId, ir), img)
		return np.array(ROI).astype(int), seg, points, img
	
	else:
		print('Glass not found')
		return None, None, None


def getExtremePoints(seg):

	contours, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#contours = contours[0]
	contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
	contours = max(contour_sizes, key=lambda x: x[0])[1]

	#The most right point
	point_r = (contours[contours[:,:,0].argmin()][0]).reshape(1,2)
	#The most left point
	point_l = (contours[contours[:,:,0].argmax()][0]).reshape(1,2)
	#The most up point
	point_u = (contours[contours[:,:,1].argmin()][0]).reshape(1,2)
	#The most down point
	point_d = (contours[contours[:,:,1].argmax()][0]).reshape(1,2)
	
	return np.concatenate((point_u, point_r, point_d, point_l), axis=0)
