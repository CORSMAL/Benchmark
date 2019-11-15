# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle
from numpy.linalg import inv

 # Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)

# Create grid board object we're using in our stream
CHARUCO_BOARD = aruco.CharucoBoard_create(squaresX=10, 
                                          squaresY=6, 
                                          squareLength=0.04,
                                          markerLength=0.03,
                                          dictionary=ARUCO_DICT)


def readCameraCalibration():
    # Check for camera calibration data
    f = open('data/calibration/Logitech/C1.pckl', 'rb')
    (cameraMatrix, distCoeffs, _, _) = pickle.load(f)
    f.close()
    if cameraMatrix is None or distCoeffs is None:
        print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
        exit()
    else:
        print('Calibration file read succesfully!')

    return cameraMatrix, distCoeffs

def getCalibrationFrame():

    cam = cv2.VideoCapture(206)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)


    fr = 0
    while(cam.isOpened()):
        ret, img = cam.read()
        if ret:
            fr += 1

        if fr == 30:
            return img
            #return cv2.undistort(img, cameraMatrix, distCoeffs)



def estimatePoseToBoard(_img, cameraMatrix, distCoeffs):

    img = _img.copy()

    # grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect Aruco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

    # Refine detected markers
    # Eliminates markers not part of our board, adds missing markers to the board
    corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(image = gray,
                                                                                board = CHARUCO_BOARD,
                                                                                detectedCorners = corners,
                                                                                detectedIds = ids,
                                                                                rejectedCorners = rejectedImgPoints,
                                                                                cameraMatrix = cameraMatrix,
                                                                                distCoeffs = distCoeffs)  

    ## REMOVE ID 49 (the robot marker)
    corners, ids = removeMarkerById(corners, ids, 49)

    img = aruco.drawDetectedMarkers(img, corners, ids=ids, borderColor=(0, 0, 255))

    rvec, tvec = None, None

    # Only try to find CharucoBoard if we found markers
    if ids is not None and len(ids) > 10:

        # Get charuco corners and ids from detected aruco markers
        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners=corners,
                                                                                markerIds=ids,
                                                                                image=gray,
                                                                                board=CHARUCO_BOARD)

        # Require more than 20 squares
        if response is not None and response > 20:
            # Estimate the posture of the charuco board
            pose, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners=charuco_corners, 
                                                            charucoIds=charuco_ids, 
                                                            board=CHARUCO_BOARD, 
                                                            cameraMatrix=cameraMatrix, 
                                                            distCoeffs=distCoeffs)


            
            img = aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 2)
            cv2.imwrite('calib_board.png', img)

    else:
        print('Calibration board is not fully visible')
        assert 1==0

    return cv2.Rodrigues(rvec)[0], tvec


def estimatePoseToMarker(_img, cameraMatrix, distCoeffs):

    img = _img.copy()

    # grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect Aruco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

    ## Keep ID 49 (the robot marker)
    corners, ids = keepMarkerById(corners, ids, 49)

    img = aruco.drawDetectedMarkers(img, corners, ids=ids, borderColor=(0, 0, 255))

    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners=corners,
                                                    markerLength=0.1965,
                                                    cameraMatrix=cameraMatrix, 
                                                    distCoeffs=distCoeffs)

    
    img = aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 2)
    cv2.imwrite('calib_marker.png', img)

    return cv2.Rodrigues(rvec)[0], tvec.reshape(3,1)

def removeMarkerById(corners, ids, id2remove):
    newCorners = []
    newIds = []
    for i in range(0,len(ids)):
        if np.asscalar(ids[i]) != id2remove:

            newCorners.append(corners[i])
            newIds.append(np.asscalar(ids[i]))

    return newCorners, np.asarray(newIds).reshape(-1,1)

def keepMarkerById(corners, ids, id2keep):
    newCorners = []
    newIds = []
    for i in range(0,len(ids)):
        if np.asscalar(ids[i]) == id2keep:

            newCorners.append(corners[i])
            newIds.append(np.asscalar(ids[i]))

    return newCorners, np.asarray(newIds).reshape(-1,1)

def getPoseFromRotationTranslation(rvec, tvec):
    C = np.concatenate((rvec, tvec), axis=1)
    return np.concatenate((C, np.array([0,0,0,1]).reshape(1,4)), axis=0)

def getRotationTranslationFromPose(C):
    rvec = C[:3,:3]
    tvec = C[:3,-1]
    return rvec, tvec



if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    # Read camera calibration
    cameraMatrix, distCoeffs = readCameraCalibration()

    # Read one frame (after discarding 30)
    img = getCalibrationFrame()
    #cv2.imwrite('robot-cameras-calibration.png', img)

    rvec_board, tvec_board = estimatePoseToBoard(img, cameraMatrix, distCoeffs)
    rvec_marker, tvec_marker = estimatePoseToMarker(img, cameraMatrix, distCoeffs)

    # Manual measurements between marker and robot
    r = np.eye(3, dtype=float)
    t = np.array([0.03, 0.25, -0.510], dtype=float).reshape(3,1)
    C_marker2robot = getPoseFromRotationTranslation(r, t)

    C_board2camera  = getPoseFromRotationTranslation(rvec_board , tvec_board )
    C_marker2camera = getPoseFromRotationTranslation(rvec_marker, tvec_marker)
    

    #TESTING
    '''
    point1, _ = cv2.projectPoints(np.array([0.2,0.,0.], dtype=float).reshape(1,3), rvec_board, tvec_board, cameraMatrix, distCoeffs)
    point2, _ = cv2.projectPoints(np.array([0.1, 0., 0.], dtype=float).reshape(1,3), rvec_marker, tvec_marker, cameraMatrix, distCoeffs)
    point1 = point1.squeeze().astype(int)
    point2 = point2.squeeze().astype(int)
    cv2.circle(img, tuple(point1), 10, (0,255,0), -1)
    cv2.circle(img, tuple(point2), 10, (255,255,0), -1)
    cv2.imwrite('test.png', img)
    assert 1==0
    '''
    #######
    
    C = np.matmul(np.matmul(C_marker2robot, inv(C_marker2camera)), C_board2camera)
    p = np.array([0.,0.,0.,1.], dtype=float).reshape(4,1)
    pr = np.matmul(C, p)
    #print(pr)

       
    # Dump projection matrix to file
    f = open('data/calibration/cameras_robot.pckl', 'wb')
    pickle.dump((C), f)
    f.close()

    print('Camera to robot transformation succesfully computed')