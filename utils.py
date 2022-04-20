import os
import numpy as np
import PIL.Image as pil
from collections import Counter

def pilLoader(imagePath):
    with open(imagePath, 'rb') as f:
        image = pil.open(f)
        image = image.convert('RGB')
        return image
    
def readCalibrationFile(calibrationFilePath):
    floatChars = set("0123456789.e+- ")
    data = {}
    with open(calibrationFilePath, "r") as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if floatChars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data

def loadVelodyneFile(velodyneFilePath):
	points = np.fromfile(velodyneFilePath, dtype=np.float32).reshape(-1, 4)
	points[:, 3] = 1.0
	return points

def subscriptsToIndices(matrixSize, rowSubscripts, colSubscripts):
	m, n = matrixSize
	return rowSubscripts * (n-1) + colSubscripts - 1

def generateDepthMap(calibrationDirectory, velodyneFilePath, cam=2, velodyneDepth=False):
    cam2cam = readCalibrationFile(os.path.join(calibrationDirectory, 'calib_cam_to_cam.txt'))
    velo2cam = readCalibrationFile(os.path.join(calibrationDirectory, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    imgShape = cam2cam["S_rect_02"][::-1].astype(np.int32)
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
    velodyne = loadVelodyneFile(velodyneFilePath)
    velodyne = velodyne[velodyne[:, 0] >= 0, :]
    velodynePointsImg = np.dot(P_velo2im, velodyne.T).T
    velodynePointsImg[:, :2] = velodynePointsImg[:, :2]/velodynePointsImg[:, 2][..., np.newaxis]
    if velodyneDepth:
        velodynePointsImg[:, 2] = velodyne[:, 0]
    velodynePointsImg[:, 0] = np.round(velodynePointsImg[:, 0]) - 1
    velodynePointsImg[:, 1] = np.round(velodynePointsImg[:, 1]) - 1
    velodyneIndices = (velodynePointsImg[:, 0] >= 0) & (velodynePointsImg[:, 1] >= 0)
    velodyneIndices = velodyneIndices & (velodynePointsImg[:, 0] < imgShape[1]) & (velodynePointsImg[:, 1] < imgShape[0])
    velodynePointsImg = velodynePointsImg[velodyneIndices, :]
    depth = np.zeros((imgShape[:2]))
    depth[velodynePointsImg[:, 1].astype(np.int), velodynePointsImg[:, 0].astype(np.int)] = velodynePointsImg[:, 2]
    indices = subscriptsToIndices(depth.shape, velodynePointsImg[:, 1], velodynePointsImg[:, 0])
    duplicateIndices = [item for item, count in Counter(indices).items() if count > 1]
    for di in duplicateIndices:
        points = np.where(indices == di)[0]
        x_loc = int(velodynePointsImg[points[0], 0])
        y_loc = int(velodynePointsImg[points[0], 1])
        depth[y_loc, x_loc] = velodynePointsImg[points, 2].min()
    depth[depth < 0] = 0
    return depth
