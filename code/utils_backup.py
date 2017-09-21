import numpy as np
import PIL.Image
import copy
import sys
import os
import cv2
import scipy.ndimage as ndimage
#import pydensecrf.densecrf as dcrf
#from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
	#create_pairwise_gaussian, unary_from_softmax
from skimage import segmentation
#from skimage.future import graph

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from layers import PlaneDepthLayer
#from layers import PlaneNormalLayer
#from html import HTML

class ColorPalette:
	def __init__(self, numColors):
		np.random.seed(1)
		self.colorMap = np.random.randint(255, size = (numColors, 3))
		self.colorMap[0] = 0
		return

	def getColorMap(self):
		return self.colorMap
	
	def getColor(self, index):
		if index >= colorMap.shape[0]:
			return np.random.randint(255, size = (3))
		else:
			return self.colorMap[index]
			pass
		

def getNYURGBDCamera():
    camera = {}
	camera['fx'] = 5.1885790117450188e+02
	camera['fy'] = 5.1946961112127485e+02
	camera['cx'] = 3.2558244941119034e+02 - 40
	camera['cy'] = 2.5373616633400465e+02 - 44
	camera['width'] = 560
	camera['height'] = 426
	return camera

def writePointCloud(filename, pointCloud, color = [255, 255, 255]):
	with open(filename, 'w') as f:
		header = """ply
format ascii 1.0
element vertex """
		header += str(pointCloud.shape[0])
		header += """
property float x
property float y
property float z
property uchar red									 { start of vertex color }
property uchar green
property uchar blue
end_header
"""
		f.write(header)
		for point in pointCloud:
			for value in point:
				f.write(str(value) + ' ')
				continue
			for value in color:
				f.write(str(value) + ' ')
				continue
			f.write('\n')
			continue
		f.close()
		pass
	return


def writeClusteringPointCloud(filename, pointCloud, clusters):
	with open(filename, 'w') as f:
		header = """ply
format ascii 1.0
element vertex """
		header += str(pointCloud.shape[0])
		header += """
property float x
property float y
property float z
property uchar red									 { start of vertex color }
property uchar green
property uchar blue
end_header
"""
		colorMap = np.random.randint(255, size = clusters.shape)
		assignment = np.argmin(np.linalg.norm(pointCloud.reshape(-1, 1, 3).repeat(clusters.shape[0], 1)[:] - clusters, 2, 2), 1)
		f.write(header)
		for pointIndex, point in enumerate(pointCloud):
			for value in point:
				f.write(str(value) + ' ')
				continue
			color = colorMap[assignment[pointIndex]]
			for value in color:
				f.write(str(value) + ' ')
				continue
			f.write('\n')
			continue
		f.close()
		pass
	return


def writeNearestNeighbors(filename, pointCloudSource, pointCloudTarget):
	with open(filename, 'w') as f:
		header = """ply
format ascii 1.0
element vertex """
		header += str((pointCloudSource.shape[0] + pointCloudTarget.shape[0] + pointCloudSource.shape[0]) * 4)
		header += """
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face """
		header += str(pointCloudSource.shape[0] + pointCloudTarget.shape[0] + pointCloudSource.shape[0])
		header += """
property list uchar int vertex_index
end_header
"""
		f.write(header)
		
		sourceColor = [0, 255, 0]
		targetColor = [0, 0, 255]
		colorMap = np.random.randint(255, size = pointCloudSource.shape)
		
		# for pointIndex, point in enumerate(pointCloudSource):
		#	 for value in point:
		#		 f.write(str(value) + ' ')
		#		 continue
		#	 color = sourceColor
		#	 for value in color:
		#		 f.write(str(value) + ' ')
		#		 continue
		#	 f.write('\n')
		#	 continue

		# for pointIndex, point in enumerate(pointCloudTarget):
		#	 for value in point:
		#		 f.write(str(value) + ' ')
		#		 continue
		#	 color = targetColor
		#	 for value in color:
		#		 f.write(str(value) + ' ')
		#		 continue
		#	 f.write('\n')
		#	 continue		

		planeSize = 0.1
		for planeType, planes in enumerate([pointCloudSource, pointCloudTarget]):
			for planeIndex, plane in enumerate(planes):
				planeD = np.linalg.norm(plane)
				planeNormal = -plane / planeD

				maxNormalDim = np.argmax(np.abs(plane))
				allDims = [0, 1, 2]
				allDims.remove(maxNormalDim)
				dim_1, dim_2 = allDims
				for delta_1, delta_2 in [(-planeSize, -planeSize), (planeSize, -planeSize), (planeSize, planeSize), (-planeSize, planeSize)]:
					point = copy.deepcopy(plane)
					point[dim_1] += delta_1
					point[dim_2] += delta_2
					point[maxNormalDim] = (-planeD - planeNormal[dim_1] * point[dim_1] - planeNormal[dim_2] * point[dim_2]) / planeNormal[maxNormalDim]

					for value in point:
						f.write(str(value) + ' ')
						continue
					if planeType == 0:
						color = sourceColor
					else:
						color = targetColor
						pass
					
					for value in color:
						f.write(str(value) + ' ')
						continue
					f.write('\n')
					continue
				continue
			continue

		assignment = np.argmin(np.linalg.norm(pointCloudSource.reshape(-1, 1, 3).repeat(pointCloudTarget.shape[0], 1)[:] - pointCloudTarget, 2, 2), 1)

		planeSize = 0.01
		lineColor = [255, 0, 0]
		for planeIndex, planeSource in enumerate(pointCloudSource):
			planeD = np.linalg.norm(planeSource)
			planeNormal = -planeSource / planeD			

			maxNormalDim = np.argmax(np.abs(planeSource))
			allDims = [0, 1, 2]
			allDims.remove(maxNormalDim)
			dim_1, dim_2 = allDims
			minNormalDim = np.argmin(np.abs(planeSource))

			for delta in [-planeSize, planeSize]:
				point = copy.deepcopy(planeSource)
				point[minNormalDim] += delta
				point[maxNormalDim] = (-planeD - planeNormal[dim_1] * point[dim_1] - planeNormal[dim_2] * point[dim_2]) / planeNormal[maxNormalDim]
				for value in point:
					f.write(str(value) + ' ')
					continue
				color = lineColor
				for value in color:
					f.write(str(value) + ' ')
					continue
				f.write('\n')
				continue

			planeTarget = pointCloudTarget[assignment[planeIndex]]
			planeDTarget = np.linalg.norm(plane)
			planeNormalTarget = -plane / planeD
			planeD = np.linalg.norm(planeTarget)
			planeNormal = -planeTarget / planeD			

			for delta in [planeSize, -planeSize]:
				point = copy.deepcopy(planeTarget)
				point[minNormalDim] += delta
				point[maxNormalDim] = (-planeD - planeNormal[dim_1] * point[dim_1] - planeNormal[dim_2] * point[dim_2]) / planeNormal[maxNormalDim]
				for value in point:
					f.write(str(value) + ' ')
					continue
				color = lineColor
				for value in color:
					f.write(str(value) + ' ')
					continue
				f.write('\n')
				continue
			continue

		for index in xrange(pointCloudSource.shape[0] + pointCloudTarget.shape[0] + pointCloudSource.shape[0]):
			planeIndex = index * 4
			f.write('4 ' + str(planeIndex + 0) + ' ' + str(planeIndex + 1) + ' ' + str(planeIndex + 2) + ' ' + str(planeIndex + 3) + '\n')
			continue

		# for pointIndexSource, point in enumerate(pointCloudSource):
	#	 pointIndexTarget = assignment[pointIndexSource]
#	 f.write(str(pointIndexSource) + ' ' + str(pointIndexTarget + pointCloudSource.shape[0]) + ' ')
		#	 color = colorMap[pointIndexSource]
	#	 for value in color:
#		 f.write(str(value) + ' ')
		#		 continue
	#	 f.write('\n')
#	 continue


		f.close()
		pass
	return


def evaluatePlanes(planes, filename = None, depths = None, normals = None, invalidMask = None, outputFolder = None, outputIndex = 0, colorMap = None):
	if filename != None:
		if 'mlt' not in filename:
			filename = filename.replace('color', 'mlt')
			pass
		normalFilename = filename.replace('mlt', 'norm_camera')
		normals = np.array(PIL.Image.open(normalFilename)).astype(np.float) / 255 * 2 - 1
		norm = np.linalg.norm(normals, 2, 2)
		for c in xrange(3):
			normals[:, :, c] /= norm
			continue
		
		depthFilename = filename.replace('mlt', 'depth')
		depths = np.array(PIL.Image.open(depthFilename)).astype(np.float) / 1000
		# if len(depths.shape) == 3:
		#	 depths = depths.mean(2)
		#	 pass
		maskFilename = filename.replace('mlt', 'valid')	
		invalidMask = np.array(PIL.Image.open(maskFilename))
		invalidMask = invalidMask < 128
		invalidMask += depths > 10
		pass

	height = normals.shape[0]
	width = normals.shape[1]
	focalLength = 517.97
	urange = np.arange(width).reshape(1, -1).repeat(height, 0) - width * 0.5
	vrange = np.arange(height).reshape(-1, 1).repeat(width, 1) - height * 0.5
	ranges = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength]).transpose([1, 2, 0])
	
	X = depths / focalLength * urange
	Y = depths
	Z = -depths / focalLength * vrange
	d = -(normals[:, :, 0] * X + normals[:, :, 1] * Y + normals[:, :, 2] * Z)	


	normalDotThreshold = np.cos(np.deg2rad(30))
	distanceThreshold = 50000
	
	reconstructedNormals = np.zeros(normals.shape)
	reconstructedDepths = np.zeros(depths.shape)
	segmentationImage = np.zeros((height, width, 3))
	distanceMap = np.ones((height, width)) * distanceThreshold
	occupancyMask = np.zeros((height, width)).astype(np.bool)
	segmentationTest = np.zeros((height, width))
	y = 297
	x = 540
	for planeIndex, plane in enumerate(planes):
		planeD = np.linalg.norm(plane)
		planeNormal = -plane / planeD

		normalXYZ = np.dot(ranges, planeNormal)
		normalXYZ = np.reciprocal(normalXYZ)
		planeY = -normalXYZ * planeD

		distance = np.abs(planeNormal[0] * X + planeNormal[1] * Y + planeNormal[2] * Z + planeD) / np.abs(np.dot(normals, planeNormal))
		#distance = np.abs(planeY - depths)
		
		mask = (distance < distanceMap) * (planeY > 0) * (np.abs(np.dot(normals, planeNormal)) > normalDotThreshold) * (np.abs(planeY - depths) < 0.5)
		occupancyMask += mask
		
		reconstructedNormals[mask] = planeNormal
		
		
		#if planeNormal[2] > 0.9:
		#print(planeD)
		#print(planeNormal)
		# minDepth = depths.min()
		# maxDepth = depths.max()
		# print(depths[300][300])
		# print(planeY[300][300])
		# print(depths[350][350])
		# print(planeY[350][350])
		# PIL.Image.fromarray((np.maximum(np.minimum((planeY - minDepth) / (maxDepth - minDepth), 1), 0) * 255).astype(np.uint8)).save(outputFolder + '/plane.png')
		# exit(1)
		#pass
		reconstructedDepths[mask] = planeY[mask]
		if colorMap != None and planeIndex in colorMap:
			segmentationImage[mask] = colorMap[planeIndex]
		else:
			segmentationImage[mask] = np.random.randint(255, size=(3,))
			pass
		distanceMap[mask] = distance[mask]
		segmentationTest[mask] = planeIndex + 1
		#print((planeIndex, planeY[y][x], distance[y][x], np.abs(np.dot(normals, planeNormal))[y][x]))
		continue

	# print(distanceMap.mean())
# print(distanceMap.max())
	# print(np.abs(reconstructedDepths - depths)[occupancyMask].max())
# print(pow(reconstructedDepths - depths, 2)[True - invalidMask].mean())
	# exit(1)

	# planeIndex = segmentationTest[y][x]
# print(normals[y][x])
	# plane = planes[int(planeIndex)]
# planeD = np.linalg.norm(plane)
	# planeNormal = -plane / planeD
# print((planeNormal, planeD))
	# print(depths[y][x])
# print(reconstructedDepths[y][x])
	# print(segmentationTest[y][x])

	if outputFolder != None:
		depths[invalidMask] = 0
		normals[invalidMask] = 0
		reconstructedDepths[invalidMask] = 0
		reconstructedNormals[invalidMask] = 0
		minDepth = depths.min()
		maxDepth = depths.max()
		#print(minDepth)
		#print(maxDepth)
		PIL.Image.fromarray(((depths - minDepth) / (maxDepth - minDepth) * 255).astype(np.uint8)).save(outputFolder + '/' + str(outputIndex) + '_depth.png')
		PIL.Image.fromarray((np.maximum(np.minimum((reconstructedDepths - minDepth) / (maxDepth - minDepth), 1), 0) * 255).astype(np.uint8)).save(outputFolder + '/' + str(outputIndex) + '_depth_reconstructed.png')
		#PIL.Image.fromarray((np.maximum(np.minimum((reconstructedDepths - depths) / (distanceThreshold), 1), 0) * 255).astype(np.uint8)).save(outputFolder + '/depth_' + str(outputIndex) + '_diff.png')
		PIL.Image.fromarray(((normals + 1) / 2 * 255).astype(np.uint8)).save(outputFolder + '/' + str(outputIndex) + '_normal_.png')
		PIL.Image.fromarray(((reconstructedNormals + 1) / 2 * 255).astype(np.uint8)).save(outputFolder + '/' + str(outputIndex) + '_normal_reconstructed.png')
		PIL.Image.fromarray(segmentationImage.astype(np.uint8)).save(outputFolder + '/' + str(outputIndex) + '_plane_segmentation.png')
		#depthImage = ((depths - minDepth) / (maxDepth - minDepth) * 255).astype(np.uint8)
		#PIL.Image.fromarray((invalidMask * 255).astype(np.uint8)).save(outputFolder + '/mask.png')
		#exit(1)
	else:
		occupancy = (occupancyMask > 0.5).astype(np.float32).sum() / (1 - invalidMask).sum()
		invalidMask += np.invert(occupancyMask)
		#PIL.Image.fromarray(invalidMask.astype(np.uint8) * 255).save(outputFolder + '/mask.png')
		reconstructedDepths = np.maximum(np.minimum(reconstructedDepths, 10), 0)
		depthError = pow(reconstructedDepths - depths, 2)[np.invert(invalidMask)].mean()
		#depthError = distanceMap.mean()
		normalError = np.arccos(np.maximum(np.minimum(np.sum(reconstructedNormals * normals, 2), 1), -1))[np.invert(invalidMask)].mean()
		#normalError = pow(np.linalg.norm(reconstructedNormals - normals, 2, 2), 2)[True - invalidMask].mean()
		#print((depthError, normalError, occupancy))
		# print(depths.max())
		# print(depths.min())
		# print(reconstructedDepths.max())
		# print(reconstructedDepths.min())
		# print(occupancy)
		# exit(1)
		
		#reconstructedDepths[np.invert(occupancyMask)] = depths[np.invert(occupancyMask)]
		return depthError, normalError, occupancy, segmentationTest, reconstructedDepths, occupancyMask
	return


def evaluatePlanesSeparately(planes, filename, outputFolder = None, outputIndex = 0):
	if 'mlt' not in filename:
		filename = filename.replace('color', 'mlt')
		pass
	colorImage = np.array(PIL.Image.open(filename))
	normalFilename = filename.replace('mlt', 'norm_camera')
	normals = np.array(PIL.Image.open(normalFilename)).astype(np.float) / 255 * 2 - 1
	height = normals.shape[0]
	width = normals.shape[1]
	norm = np.linalg.norm(normals, 2, 2)
	for c in xrange(3):
		normals[:, :, c] /= norm
		continue

	
	depthFilename = filename.replace('mlt', 'depth')
	depths = np.array(PIL.Image.open(depthFilename)).astype(np.float) / 1000
	# if len(depths.shape) == 3:
	#	 depths = depths.mean(2)
	#	 pass
	
	focalLength = 517.97
	urange = np.arange(width).reshape(1, -1).repeat(height, 0) - width * 0.5
	vrange = np.arange(height).reshape(-1, 1).repeat(width, 1) - height * 0.5
	ranges = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength]).transpose([1, 2, 0])
	X = depths / focalLength * urange
	Y = depths
	Z = -depths / focalLength * vrange
	d = -(normals[:, :, 0] * X + normals[:, :, 1] * Y + normals[:, :, 2] * Z)

	
	maskFilename = filename.replace('mlt', 'valid')	
	invalidMask = np.array(PIL.Image.open(maskFilename))
	# if len(invalidMask.shape) == 3:
	#	 invalidMask = invalidMask.mean(2)
	#	 pass
	invalidMask = invalidMask < 128
	invalidMask += depths > 10


	normalDotThreshold = np.cos(np.deg2rad(15))
	distanceThreshold = 0.15
	colorPalette = ColorPalette(len(planes))
	for planeIndex, plane in enumerate(planes):
		planeD = np.linalg.norm(plane)
		planeNormal = -plane / planeD

		distance = np.abs(planeNormal[0] * X + planeNormal[1] * Y + planeNormal[2] * Z + planeD)

		normalXYZ = np.dot(ranges, planeNormal)
		normalXYZ = np.reciprocal(normalXYZ)
		planeY = -normalXYZ * planeD
		
		mask = (planeY > 0) * (np.abs(np.dot(normals, planeNormal)) > normalDotThreshold) * (distance < distanceThreshold)

		maxDepth = 10
		minDepth = 0
		#PIL.Image.fromarray((np.minimum(np.maximum((planeY - minDepth) / (maxDepth - minDepth), 0), 1) * 255).astype(np.uint8)).save(outputFolder + '/plane_depth_' + str(planeIndex) + '.png')
		#PIL.Image.fromarray(((planeNormal.reshape(1, 1, 3).repeat(height, 0).repeat(width, 1) + 1) / 2 * 255).astype(np.uint8)).save(outputFolder + '/plane_normal_' + str(planeIndex) + '.png')
		planeImage = colorImage * 0.3
		planeImage[mask] += colorPalette.getColor(planeIndex) * 0.7
		PIL.Image.fromarray(planeImage.astype(np.uint8)).save(outputFolder + '/plane_mask_' + str(planeIndex) + '_' + str(outputIndex) + '.png')
		#PIL.Image.fromarray(mask.astype(np.uint8) * 255).save(outputFolder + '/mask_' + str(planeIndex) + '.png')
		continue
	return

def residual2Planes(residualPlanes, predefinedPlanes):
	numClusters = predefinedPlanes.shape[0]
	planes = []
	for residualPlane in residualPlanes:
		gridIndex = int(residualPlane[0]) / numClusters
		planeIndex = int(residualPlane[0]) % numClusters
		planes.append(predefinedPlanes[planeIndex] + residualPlane[1:])
		continue
	return planes

def residual2PlanesGlobal(residualPlanes, predefinedPlanes):
	numClusters = predefinedPlanes.shape[0]
	planes = []
	for residualPlane in residualPlanes:
		planeIndex = int(residualPlane[0])
		planes.append(predefinedPlanes[planeIndex] + residualPlane[1:])
		continue
	return planes


def getPlaneInfo(planes):
    imageWidth = 640
	imageHeight = 480
	focalLength = 517.97
	urange = np.arange(imageWidth).reshape(1, -1).repeat(imageHeight, 0) - imageWidth * 0.5
	vrange = np.arange(imageHeight).reshape(-1, 1).repeat(imageWidth, 1) - imageHeight * 0.5
	ranges = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength]).transpose([1, 2, 0])
	
	planeDepths = PlaneDepthLayer(planes, ranges)
	planeNormals = PlaneNormalLayer(planes, ranges)
	return planeDepths, planeNormals

def getProbability(image, segmentation):
    width = image.shape[1]
	height = image.shape[0]
	numPlanes = segmentation.shape[0]
	probabilities = np.exp(segmentation)
	probabilities = probabilities / probabilities.sum(0)
	# The input should be the negative of the logarithm of probability values
	# Look up the definition of the softmax_to_unary for more information
	unary = unary_from_softmax(probabilities)

	# The inputs should be C-continious -- we are using Cython wrapper
	unary = np.ascontiguousarray(unary)

	d = dcrf.DenseCRF(height * width, numPlanes)
	
	d.setUnaryEnergy(unary)

	# This potential penalizes small pieces of segmentation that are
	# spatially isolated -- enforces more spatially consistent segmentations
	# feats = create_pairwise_gaussian(sdims=(10, 10), shape=(height, width))
	# d.addPairwiseEnergy(feats, compat=300,
	#										 kernel=dcrf.DIAG_KERNEL,
	#										 normalization=dcrf.NORMALIZE_SYMMETRIC)

		# This creates the color-dependent features --
		# because the segmentation that we get from CNN are too coarse
		# and we can use local color features to refine them
		
		feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
										  img=image, chdim=2)
		d.addPairwiseEnergy(feats, compat=10,
							kernel=dcrf.DIAG_KERNEL,
							normalization=dcrf.NORMALIZE_SYMMETRIC)
		
		Q = d.inference(50)

		inds = np.argmax(Q, axis=0).reshape((height, width))
		probabilities = np.zeros((height * width, numPlanes))
		probabilities[np.arange(height * width), inds.reshape(-1)] = 1
		probabilities = probabilities.reshape([height, width, -1])
		#print(res.shape)
		return probabilities

def getProbabilityMax(segmentation):
	width = segmentation.shape[2]
	height = segmentation.shape[1]
	numPlanes = segmentation.shape[0]
	inds = np.argmax(segmentation.reshape([-1, height * width]), axis=0)
	probabilities = np.zeros((height * width, numPlanes))
	probabilities[np.arange(height * width), inds] = 1
	probabilities = probabilities.reshape([height, width, -1])
		return probabilities
	
def evaluateSegmentation(testdir, image, depth, normal, segmentation, planes, resultIndex=0):
	width = depth.shape[1]
	height = depth.shape[0]
	numPlanes = planes.shape[0]
	planeDepths, planeNormals = getPlaneInfo(planes)
	probabilities_small = getProbabilityMax(segmentation)
	probabilities = np.zeros(planeDepths.shape)

		for index in xrange(probabilities.shape[2]):
			probabilities[:, :, index] = cv2.resize(probabilities_small[:, :, index], (width, height), interpolation=cv2.INTER_LINEAR)
				continue

		depthPred = (probabilities * planeDepths).sum(2)
		normalPred = (probabilities.reshape([height, width, numPlanes, 1]).repeat(3, 3) * planeNormals).sum(2)
		#depthPred, normalPred = drawDepthNormal(depth.shape[1], depth.shape[0], segmentation, planes)

		randomColor = np.random.randint(255, size=(numPlanes, 3))
		cv2.imwrite(testdir + '/' + str(resultIndex) + '_image.png', image)
		cv2.imwrite(testdir + '/' + str(resultIndex) + '_depth.png', np.minimum(np.maximum(depth / 10 * 255, 0), 255).astype(np.uint8))
		cv2.imwrite(testdir + '/' + str(resultIndex) + '_output_depth.png', np.minimum(np.maximum(depthPred / 10 * 255, 0), 255).astype(np.uint8))
		cv2.imwrite(testdir + '/' + str(resultIndex) + '_normal.png', ((normal + 1) / 2 * 255).astype(np.uint8))
		cv2.imwrite(testdir + '/' + str(resultIndex) + '_output_normal.png', ((normalPred + 1) / 2 * 255).astype(np.uint8))
		#cv2.imwrite(testdir + '/' + str(resultIndex) + '_segmentation.png', np.dot(probabilities, randomColor).astype(np.uint8))
		cv2.imwrite(testdir + '/' + str(resultIndex) + '_output_segmentation.png', np.dot(probabilities, randomColor).astype(np.uint8))

		return
	probabilities_small = getProbability(cv2.resize(depth.reshape(height, width, 1).repeat(3, 2), (segmentation.shape[2], segmentation.shape[1])), segmentation)
	probabilities = np.zeros(planeDepths.shape)
		for index in xrange(probabilities.shape[2]):
			probabilities[:, :, index] = cv2.resize(probabilities_small[:, :, index], (width, height), interpolation=cv2.INTER_LINEAR)
				continue

		depthPred = (probabilities * planeDepths).sum(2)
		normalPred = (probabilities.reshape([height, width, numPlanes, 1]).repeat(3, 3) * planeNormals).sum(2)
		#depthPred, normalPred = drawDepthNormal(depth.shape[1], depth.shape[0], segmentation, planes)
		

		cv2.imwrite(testdir + '/' + str(resultIndex) + '_output_depth_mrf.png', np.minimum(np.maximum(depthPred / 10 * 255, 0), 255).astype(np.uint8))
		cv2.imwrite(testdir + '/' + str(resultIndex) + '_output_normal_mrf.png', ((normalPred + 1) / 2 * 255).astype(np.uint8))
		cv2.imwrite(testdir + '/' + str(resultIndex) + '_output_segmentation_mrf.png', np.dot(probabilities, np.random.randint(255, size=(numPlanes, 3))).astype(np.uint8))

		return


def calcPlaneInfo(im_name):
	normalFilename = im_name['normal']
	normals = np.array(PIL.Image.open(normalFilename)).astype(np.float) / 255 * 2 - 1
	norm = np.linalg.norm(normals, 2, 2)
		for c in xrange(3):
			normals[:, :, c] /= norm
				continue
			width = normals.shape[1]
			height = normals.shape[0]
			
		maskFilename = im_name['valid']
		invalidMask = (np.array(PIL.Image.open(maskFilename)) < 128)

		sampleRatio = 3
		azimuthAngleImage = (-np.round(np.rad2deg(np.arctan2(normals[:, :, 1], normals[:, :, 0])) / sampleRatio).astype(np.int) * sampleRatio + 360) % 360

		altitudeAngleImage = (np.round(np.rad2deg(np.arctan2(np.sign(-normals[:, :, 1]) * np.linalg.norm(normals[:, :, :2], 2, 2), normals[:, :, 2])) / sampleRatio).astype(np.int) * sampleRatio + 360) % 360

		orthogonalThreshold = 5
		orthogonalAzimuthMask_1 = ((azimuthAngleImage - 0) < orthogonalThreshold) + ((360 - azimuthAngleImage) < orthogonalThreshold)
		orthogonalAzimuthMask_2 = np.abs(azimuthAngleImage - 180) < orthogonalThreshold
		azimuthAngleImage[orthogonalAzimuthMask_1] = 0
		azimuthAngleImage[orthogonalAzimuthMask_2] = 180
		altitudeAngleImage[orthogonalAzimuthMask_1 + orthogonalAzimuthMask_2] = 0

		orthogonalAltitudeMask_1 = ((altitudeAngleImage - 0) < orthogonalThreshold) + ((360 - altitudeAngleImage) < orthogonalThreshold)
		orthogonalAltitudeMask_2 = np.abs(altitudeAngleImage - 180) < orthogonalThreshold
		altitudeAngleImage[orthogonalAltitudeMask_1] = 0
		altitudeAngleImage[orthogonalAltitudeMask_2] = 180
		azimuthAngleImage[orthogonalAltitudeMask_1 + orthogonalAltitudeMask_2] = 0

		azimuthAngleImage[invalidMask] = 360
		altitudeAngleImage[invalidMask] = 360
		
		
		sampleRatio = 5
		depthFilename = im_name['depth']
		depths = np.array(PIL.Image.open(depthFilename)).astype(np.float) / 1000
		focalLength = 517.97
		urange = np.arange(width).reshape(1, -1).repeat(height, 0) - width * 0.5
		vrange = np.arange(height).reshape(-1, 1).repeat(width, 1) - height * 0.5
		X = depths / focalLength * urange
		Y = depths
		Z = -depths / focalLength * vrange
		d = -(normals[:, :, 0] * X + normals[:, :, 1] * Y + normals[:, :, 2] * Z)
		dImage = np.round(d / (10. / 360) / sampleRatio).astype(np.int) * sampleRatio
		dImage[dImage < 0] = 0
		dImage[dImage > 360] = 360
		dImage[invalidMask] = 360		

		valueMaps = [azimuthAngleImage, altitudeAngleImage, dImage]
		planes = []
		values_1, counts_1 = np.unique(valueMaps[0], return_counts=True)

		globalMask = np.zeros((height, width), np.bool)
		planeAreaThreshold = 40 * 30
		segmentations = []
		for index_1, value_1 in enumerate(values_1):
				if counts_1[index_1] < planeAreaThreshold or value_1 == 360:
						continue
					mask_1 = valueMaps[0] == value_1

				values_2, counts_2 = np.unique(valueMaps[1][mask_1], return_counts=True)
				for index_2, value_2 in enumerate(values_2):
						if counts_2[index_2] < planeAreaThreshold or value_2 == 360:
								continue
							mask_2 = mask_1 * (valueMaps[1] == value_2)
							values_3, counts_3 = np.unique(valueMaps[2][mask_2], return_counts=True)
						for index_3, value_3 in enumerate(values_3):
								if counts_3[index_3] < planeAreaThreshold or value_3 == 360:
										continue
									mask_3 = mask_2 * (valueMaps[2] == value_3)
									mask_3 = ndimage.binary_erosion(mask_3).astype(mask_3.dtype)
								if mask_3.sum() < planeAreaThreshold:
										continue

								# regionX = X[mask_3]
							# regionY = Y[mask_3]
						# regionZ = Y[mask_3]
					
								normal = np.array([normals[:, :, 0][mask_3].mean(), normals[:, :, 1][mask_3].mean(), normals[:, :, 2][mask_3].mean()])
								normal /= np.linalg.norm(normal, 2)
								dPlane = (-(normal[0] * X + normal[1] * Y + normal[2] * Z))[mask_3].mean()

								globalMask += mask_3
								segmentations.append(mask_3)
								azimuth = np.arctan2(-normal[1], normal[0])
								altitude = np.arctan2(np.sign(-normal[1]) * np.linalg.norm(normal[:2]), normal[2])
								#planes.append(((azimuth, altitude, dPlane), mask_3))
								planes.append(((-normal[0] * dPlane, -normal[1] * dPlane, -normal[2] * dPlane), mask_3))


								if False:
									plane = np.array([-normal[0] * dPlane, -normal[1] * dPlane, -normal[2] * dPlane])

										#focalLength = 517.97
										#urange = np.arange(width).reshape(1, -1).repeat(height, 0) - width * 0.5
										#vrange = np.arange(height).reshape(-1, 1).repeat(width, 1) - height * 0.5
										ranges = np.array([urange / focalLength / width * 640, np.ones(urange.shape), -vrange / focalLength / height * 480]).transpose([1, 2, 0])

										#ranges = np.stack([urange, np.ones([height, width]), -vrange], axis=2)
										ranges = np.reshape(ranges, [-1, 3])
										
										planeD = np.linalg.norm(plane)
										#planeD = np.clip(planeD, 1e-5, 10)
										planeNormal = -plane / planeD

										normalXYZ = np.matmul(ranges, planeNormal.reshape(3, 1))
										#normalXYZ = np.multiply(np.sign(normalXYZ), np.clip(np.abs(normalXYZ), 1e-4, 1000000))
										normalXYZ[normalXYZ == 0] = 1e-5
										normalXYZ = 1 / normalXYZ
										planeDepth = -normalXYZ * planeD
										planeDepth = np.reshape(planeDepth, [height, width])

										planeDepth = np.clip(planeDepth, 0, 10)
										normalDotThreshold = np.cos(np.deg2rad(15))
										depthDiffThreshold = 0.1
										#print(planeDepth[mask_3])
										#print(depths[mask_3])
										mask = (np.abs(np.dot(normals, planeNormal)) > normalDotThreshold) * (np.abs(planeDepth - depths) < depthDiffThreshold)
										#mask = np.abs(np.dot(normals, planeNormal)) > normalDotThreshold
										cv2.imwrite('test_evaluate/segmentation_0_' + str(len(planes) - 1) + '_mask.png', mask.astype(np.uint8) * 255)
										cv2.imwrite('test_evaluate/segmentation_0_' + str(len(planes) - 1) + '_mask_gt.png', mask_3.astype(np.uint8) * 255)
										#exit(1)
								continue
						continue
				continue
			segmentations = np.array(segmentations)
			
		if False:
			planeParameters = np.array([plane[0] for plane in planes])
			planeD = np.linalg.norm(planeParameters, 2, 1, keepdims=True)
			planeNormals = planeParameters / planeD
			
				normalXYZ = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength])

				normalXYZ = np.dot(normalXYZ.transpose([1, 2, 0]), planeNormals.transpose())
				normalXYZ = np.reciprocal(normalXYZ)

				XYZ = np.array([X, Y, Z])
				planeXYZ = np.zeros(XYZ.shape)
				for i in xrange(planeParameters.shape[0]):
					mask = planes[i][1]
					planeY = normalXYZ[:, :, i] * planeD[i]
					planeX = planeY * urange / focalLength
					planeZ = -planeY * vrange / focalLength

						planeXYZ[0][mask] = planeX[mask]
						planeXYZ[1][mask] = planeY[mask]
						planeXYZ[2][mask] = planeZ[mask]
						continue

				# for c in xrange(3):
			#		 inputImage = XYZ[c]
		#		 cMin = inputImage.min()
	#		 cMax = inputImage.max()
#		 #PIL.Image.fromarray(((inputImage - cMin) / (cMax - cMin) * 255).astype(np.uint8)).save('test/' + str(c) + '.png')
				#		 reconstructed = planeXYZ[c]
			#		 #PIL.Image.fromarray(((reconstructed - cMin) / (cMax - cMin) * 255).astype(np.uint8)).save('test/' + str(c) + '_reconstructed.png')
		#		 continue

				planeImage = np.zeros((height, width, 3))
				for plane in planes:
					mask = plane[1]
						for c in xrange(3):
							planeImage[:, :, c][mask] = np.random.randint(0, 255)
							#planeImage[:, :, c][mask] = max(min(round((plane[0][c] + 1) / 2 * 255), 255), 0)
								continue
						continue
					cv2.imwrite('test_evaluate/plane_image.png', cv2.imread(im_name['image']))
					PIL.Image.fromarray(planeImage.astype(np.uint8)).save('test_evaluate/plane.png')
					print(planeParameters)
					#print(np.load(im_name['plane'])[:, :3])
				pass
		return globalMask, segmentations
	

def findCornerPoints(im_name, planes, segmentations = None):
	imageWidth = 640
	imageHeight = 480
	focalLength = 517.97
	urange = np.arange(imageWidth).reshape(1, -1).repeat(imageHeight, 0) - imageWidth * 0.5
	vrange = np.arange(imageHeight).reshape(-1, 1).repeat(imageWidth, 1) - imageHeight * 0.5
	ranges = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength]).transpose([1, 2, 0])
	

		normalFilename = im_name['normal']
		normal = np.array(PIL.Image.open(normalFilename)).astype(np.float) / 255 * 2 - 1
		norm = np.linalg.norm(normal, 2, 2)
		for c in xrange(3):
			normal[:, :, c] /= norm
				continue
			
		maskFilename = im_name['valid']
		invalidMask = (np.array(PIL.Image.open(maskFilename)) < 128)

		depthFilename = im_name['depth']
		depth = np.array(PIL.Image.open(depthFilename)).astype(np.float) / 1000
		depth = np.clip(depth, 0, 10)
		
		planeDepths = PlaneDepthLayer(planes, ranges)
		
		depthDiffThreshold = 0.15
		occludedRatioThreshold = 0.1
		occludedMasks = (planeDepths - np.expand_dims(depth, -1)) < -depthDiffThreshold
		occludedRatio = occludedMasks.astype(np.float).mean(0).mean(0)
		planeCandidates = planes[occludedRatio < occludedRatioThreshold]
		planeConfidence = 1 - occludedRatio[occludedRatio < occludedRatioThreshold]
		planeCandidates = np.concatenate([planeCandidates, planeConfidence.reshape(-1, 1)], 1)

		angleDotThreshold = np.cos(np.deg2rad(20))
		ceilingPlaneInds = np.dot(planeCandidates[:, :3], np.array([0, 0, -1])) / np.linalg.norm(planeCandidates[:, :3], 2, 1) > angleDotThreshold
		ceilingPlanes = planeCandidates[ceilingPlaneInds]
		if ceilingPlanes.shape[0] > 1:
			planeConfidence = ceilingPlanes[:, 3:].sum(1)
			sortInds = np.argsort(planeConfidence)[::-1]
			ceilingPlanes = ceilingPlanes[sortInds[:1]]
				pass
			#print(np.dot(planeCandidates[:, :3], np.array([0, 0, -1])) / np.linalg.norm(planeCandidates[:, :3], 2, 1))
		#print(ceilingPlanes)

		horizontalPlanes = {}
		if ceilingPlanes.shape[0] > 0:
			horizontalPlanes['ceiling'] = ceilingPlanes[0, :3]
				pass
			
		floorPlaneInds = np.dot(planeCandidates[:, :3], np.array([0, 0, 1])) / np.linalg.norm(planeCandidates[:, :3], 2, 1) > angleDotThreshold
		floorPlanes = planeCandidates[floorPlaneInds]
		floorPlanes = planeCandidates[floorPlaneInds]
		if floorPlanes.shape[0] > 1:
			planeConfidence = floorPlanes[:, 3:].sum(1)
			sortInds = np.argsort(planeConfidence)[::-1]
			floorPlanes = floorPlanes[sortInds[:1]]
				pass

		if floorPlanes.shape[0] > 0:
			horizontalPlanes['floor'] = floorPlanes[0, :3]
				pass
		if len(horizontalPlanes) == 0:
				return []
			
		angleDotThreshold = np.cos(np.deg2rad(70))
		wallPlaneInds = np.abs(np.dot(planeCandidates[:, :3], np.array([0, 0, 1]))) / np.linalg.norm(planeCandidates[:, :3], 2, 1) < angleDotThreshold
		wallPlanes = planeCandidates[wallPlaneInds]
		wallPlanes = planeCandidates[wallPlaneInds]
		if wallPlanes.shape[0] > 1:
			planeConfidence = wallPlanes[:, 3:].sum(1)
			sortInds = np.argsort(planeConfidence)[::-1]
			wallPlanes = wallPlanes[sortInds]
				pass

		angles = np.arctan2(wallPlanes[:, 1], wallPlanes[:, 0])
		walls = []
		angleDiffThreshold = np.deg2rad(60)
		for wallIndex in xrange(wallPlanes.shape[0]):
				if len(walls) == 0:
					walls.append((wallPlanes[wallIndex, :3], angles[wallIndex]))
				elif len(walls) == 1:
						if abs(angles[wallIndex] - walls[0][1]) > angleDiffThreshold:
							walls.append((wallPlanes[wallIndex, :3], angles[wallIndex]))
								pass
						pass
				elif len(walls) == 2:
						if abs(angles[wallIndex] - walls[0][1]) > angleDiffThreshold and abs(angles[wallIndex] - walls[1][1]) > angleDiffThreshold:
							walls.append((wallPlanes[wallIndex, :3], angles[wallIndex]))
								pass
						pass
				else:
						break
				continue

		if len(walls) > 1:
			sortInds = np.argsort([wall[1] for wall in walls])
			walls = [walls[index][0] for index in sortInds]
		else:
				return []

		cornerPoints = []
		for wallIndex in xrange(len(walls) - 1):
			cornerPlanes = []
			cornerPlanes.append(walls[wallIndex])
			cornerPlanes.append(walls[wallIndex + 1])
				for _, plane in horizontalPlanes.items():
					cornerPlanes.append(plane)
					A = np.array(cornerPlanes)
					b = np.linalg.norm(A, 2, 1)
					A /= b.reshape(-1, 1)
					cornerPoint = np.linalg.solve(A, b)
					u = cornerPoint[0] / cornerPoint[1] * focalLength + imageWidth / 2
					v = -cornerPoint[2] / cornerPoint[1] * focalLength + imageHeight / 2
					cornerPoints.append([u, v])
					cornerPlanes.pop()
						pass
				continue
		return cornerPoints
	

def evaluateDepths(predDepths, gtDepths, validMasks, planeMasks=True, printInfo=True):
	masks = np.logical_and(np.logical_and(validMasks, planeMasks), gtDepths > 1e-4)
	
		numPixels = float(masks.sum())
		rms = np.sqrt((pow(predDepths - gtDepths, 2) * masks).sum() / numPixels)
		#log10 = (np.abs(np.log10(np.maximum(predDepths, 1e-4)) - np.log10(np.maximum(gtDepths, 1e-4))) * masks).sum() / numPixels
		#rel = (np.abs(predDepths - gtDepths) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
		deltas = np.maximum(predDepths / np.maximum(gtDepths, 1e-4), gtDepths / np.maximum(predDepths, 1e-4)) + (1 - masks.astype(np.float32)) * 10000
		accuracy_1 = (deltas < 1.25).sum() / numPixels
		accuracy_2 = (deltas < pow(1.25, 2)).sum() / numPixels
		accuracy_3 = (deltas < pow(1.25, 3)).sum() / numPixels
		recall = float(masks.sum()) / validMasks.sum()
		#print((rms, recall))
		if printInfo:
			print(('evaluate', rms, accuracy_1, accuracy_2, accuracy_3, recall))
				pass
		return rms, accuracy_1
	#return rel, log10, rms, accuracy_1, accuracy_2, accuracy_3, recall

def drawDepthImage(depth):
	#return cv2.applyColorMap(np.clip(depth / 10 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
		return 255 - np.clip(depth / 10 * 255, 0, 255).astype(np.uint8)
	
def drawNormalImage(normal):
		return ((normal + 1) / 2 * 255).astype(np.uint8)
	
def drawSegmentationImage(segmentations, randomColor=None, numColors=22, planeMask=1, offset=1, black=False):
	randomColor = ColorPalette(numColors).getColorMap()
		if black:
			randomColor[0] = 0
				pass
			width = segmentations.shape[1]
			height = segmentations.shape[0]
		if segmentations.ndim == 2:
			segmentation = (segmentations + offset) * planeMask
		else:
			#segmentation = (np.argmax(segmentations, 2) + 1) * (np.max(segmentations, 2) > 0.5)
				if black:
					segmentation = np.argmax(segmentations, 2)
				else:
					segmentation = (np.argmax(segmentations, 2) + 1) * planeMask
						pass
				pass
			segmentation = segmentation.astype(np.int)
		return randomColor[segmentation.reshape(-1)].reshape((height, width, 3))

def drawMaskImage(mask):
		return (mask * 255).astype(np.uint8)

def drawDiffImage(values_1, values_2, threshold):
	#return cv2.applyColorMap(np.clip(np.abs(values_1 - values_2) / threshold * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
		return np.clip(np.abs(values_1 - values_2) / threshold * 255, 0, 255).astype(np.uint8)


def getSuperpixels(depth, normal, width, height, numPlanes=50, numGlobalPlanes = 10):
	depth = np.expand_dims(depth, -1)

	urange = (np.arange(width, dtype=np.float32) / (width + 1) - 0.5) / focalLength * 641
	urange = np.tile(np.reshape(urange, [1, -1]), [height, 1])
	vrange = (np.arange(height, dtype=np.float32) / (height + 1) - 0.5) / focalLength * 481
	vrange = np.tile(np.reshape(vrange, [-1, 1]), [1, width])
	
		ranges = np.stack([urange, np.ones([height, width]), -vrange], axis=2)
		#ranges = np.expand_dims(ranges, 0)

		planeImage = np.sum(normal * ranges, axis=2, keepdims=True) * depth * normal
		planeImage = planeImage / 10 * 1000

		superpixels = segmentation.slic(planeImage, compactness=30, n_segments=400)
		g = graph.rag_mean_color(planeImage, superpixels, mode='similarity')
		planeSegmentation = graph.cut_normalized(superpixels, g)
		return planeSegmentation, superpixels


def calcPlaneDepths(planes, width, height):
	focalLength = 517.97
	urange = np.arange(width).reshape(1, -1).repeat(height, 0) - width * 0.5
	vrange = np.arange(height).reshape(-1, 1).repeat(width, 1) - height * 0.5
	ranges = np.array([urange / width * 640 / focalLength, np.ones(urange.shape), -vrange / height * 480 / focalLength]).transpose([1, 2, 0])
	
		planeDepths = PlaneDepthLayer(planes, ranges)
		return planeDepths


def writePLYFileParts(folder, index, image, depth, segmentation):
	
		imageFilename = str(index) + '_image.png'
		cv2.imwrite(folder + '/' + imageFilename, image)
		focalLength = 517.97
		width = image.shape[1]
		height = image.shape[0]
		
		for segmentIndex in xrange(segmentation.max() + 1):
			faces = []
				for y in xrange(height - 1):
						for x in xrange(width - 1):
								if segmentation[y][x] != segmentIndex:
										continue
								if segmentation[y + 1][x] == segmentIndex and segmentation[y + 1][x + 1] == segmentIndex:
									depths = [depth[y][x], depth[y + 1][x], depth[y + 1][x + 1]]
										if min(depths) > 0 and max(depths) < 10:
											faces.append((x, y, x, y + 1, x + 1, y + 1))
												pass
										pass
								if segmentation[y][x + 1] == segmentIndex and segmentation[y + 1][x + 1] == segmentIndex:
									depths = [depth[y][x], depth[y][x + 1], depth[y + 1][x + 1]]
										if min(depths) > 0 and max(depths) < 10:
											faces.append((x, y, x + 1, y + 1, x + 1, y))
												pass
										pass
								continue
						continue

				#print(len(faces))
				with open(folder + '/' + str(index) + '_model_' + str(segmentIndex) + '.ply', 'w') as f:
					header = """ply
format ascii 1.0
comment VCGLIB generated
comment TextureFile """
					header += imageFilename
					header += """
element vertex """
					header += str(width * height)
					header += """
property float x
property float y
property float z
element face """
					header += str(len(faces))
					header += """
property list uchar int vertex_indices
property list uchar float texcoord
end_header
"""
					f.write(header)
						for y in xrange(height):
								for x in xrange(width):
										if segmentation[y][x] != segmentIndex:
											f.write("0.0 0.0 0.0\n")
												continue
											Y = depth[y][x]
											X = Y / focalLength * (x - width / 2) / width * 640
											Z = -Y / focalLength * (y - height / 2) / height * 480
											f.write(str(X) + ' ' +	str(Z) + ' ' + str(-Y) + '\n')
										continue
								continue


						for face in faces:
							f.write('3 ')
								for c in xrange(3):
									f.write(str(face[c * 2 + 1] * width + face[c * 2]) + ' ')
										continue
									f.write('6 ')						
								for c in xrange(3):
									f.write(str(float(face[c * 2]) / width) + ' ' + str(1 - float(face[c * 2 + 1]) / height) + ' ')
										continue
									f.write('\n')
								continue
							f.close()
						pass
				continue
		return	

def writeHTML(folder, numImages):
	return
	h = HTML('html')
	h.p('Results')
	h.br()
	#suffixes = ['', '_crf_1']
	#folders = ['test_all_resnet_v2' + suffix + '/' for suffix in suffixes]
	for index in xrange(numImages):
		t = h.table(border='1')
		r_inp = t.tr()
		r_inp.td('input')
		r_inp.td().img(src=str(index) + '_image.png')
		r_inp.td().img(src='one.png')
		r_inp.td().img(src='one.png')
		#r_inp.td().img(src='one.png')
		r_inp.td().img(src=str(index) + '_model.png')

		r_gt = t.tr()
		r_gt.td('gt')
		r_gt.td().img(src=str(index) + '_segmentation_gt.png')
		r_gt.td().img(src=str(index) + '_depth.png')
		r_gt.td().img(src='one.png')
		r_gt.td().img(src=str(index) + '_normal.png')
				
		#r_gt.td().img(src=folders[0] + str(index) + '_depth_gt.png')
		#r_gt.td().img(src=folders[0] + '_depth_gt_diff.png')
		#r_gt.td().img(src=folders[0] + str(index) + '_normal_gt.png')

		r_pred = t.tr()
		r_pred.td('pred')
		r_pred.td().img(src=str(index) + '_segmentation_pred.png')
		r_pred.td().img(src=str(index) + '_depth_pred.png')
		r_pred.td().img(src=str(index) + '_depth_pred_diff.png')
		r_pred.td().img(src=str(index) + '_normal_pred.png')

		h.br()
		continue

	html_file = open(folder + '/index.html', 'w')
	html_file.write(str(h))
	html_file.close()
	return


def fitPlane(points):
		return np.linalg.solve(points, np.ones(points.shape[0]))

def fitPlanes(depth, numPlanes=50, planeAreaThreshold=3*4, numIterations=50, distanceThreshold=0.05):
	width = depth.shape[0]
	height = depth.shape[1]

	camera = getNYURGBDCamera()
	urange = (np.arange(width, dtype=np.float32) / width * camera['width'] - camera['cx']) / camera['fx']
	vrange = (np.arange(height, dtype=np.float32) / height * camera['height'] - camera['cy']) / camera['fy']

	X = depth / urange
	Y = depth
	Z = -depth / vrange
	XYZ = np.stack([X, Y, Z], axis=2).reshape(-1, 3)

	planes = []
	planePointsArray = []
	for planeIndex in xrange(numPlanes):
		maxNumInliers = planeAreaThreshold
		for iteration in xrange(numIterations):
		    sampledPoints = XYZ[np.random.choice(np.arange(XYZ.shape[0]), size=(3))]
			plane = fitPlane(sampledPoints)
			numInliers = np.sum(np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) < distanceThreshold)
			if numInliers > maxNumInliers:
				maxNumInliers = numInliers
				bestPlane = plane
				pass
			continue
		if maxNumInliers == planeAreaThreshold:
			break
		planes.append(bestPlane)
		inlierInds = np.abs(np.matmul(XYZ, bestPlane) - np.ones(XYZ.shape[0])) < distanceThreshold
		inliersPoints = XYZ[inlierInds]
		planePointsArray = inliersPoints
		XYZ = XYZ[logical_not(inlierInds)]
		continue
	planes = np.array(planes)
	
	planeSegmentation = np.zeros(depth.shape).reshape(-1)
	for planeIndex, planePoints in enumerate(planePointsArray):
		planeDepth = planePoints[:, 1]
		u = int((planePoints[:, 0] / planeDepth * camera['fx'] + camera['cx']) / camera['width'] * width)
		v = int((-planePoints[:, 2] / planeDepth * camera['fy'] + camera['cy']) / camera['height'] * height)
		planeSegmentation[v, u] = planeIndex
		continue
	return planes, planeSegmentation

		
