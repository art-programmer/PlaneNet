import tensorflow as tf
import numpy as np

# def segmentationRefinementModule(segmentation, planeDepths, numOutputPlanes = 20, gpu_id = 0, coef = [1, 1, 1], beta = 10):
#     with tf.device('/gpu:%d'%gpu_id):
#         S = segmentation
#         #S = tf.one_hot(tf.argmax(S, 3), numOutputPlanes)
#         D = tf.tile(tf.expand_dims(planeDepths, -1), [1, 1, 1, 1, numOutputPlanes])
#         D_transpose = tf.tile(tf.expand_dims(planeDepths, 3), [1, 1, 1, numOutputPlanes, 1])
#         D_diff = tf.abs(D - D_transpose)
#         batchSize = int(segmentation.shape[0])
#         height = int(segmentation.shape[1])
#         width = int(segmentation.shape[2])
#         S_neighbor_up = tf.concat([tf.zeros([batchSize, 1, width, numOutputPlanes]), tf.slice(S, [0, 0, 0, 0], [batchSize, height - 1, width, numOutputPlanes])], axis = 1)
#         S_neighbor_down = tf.concat([tf.slice(S, [0, 1, 0, 0], [batchSize, height - 1, width, numOutputPlanes]), tf.zeros([batchSize, 1, width, numOutputPlanes]), ], axis = 1)
#         S_neighbor_left = tf.concat([tf.zeros([batchSize, height, 1, numOutputPlanes]), tf.slice(S, [0, 0, 0, 0], [batchSize, height, width - 1, numOutputPlanes])], axis = 2)
#         S_neighbor_right = tf.concat([tf.slice(S, [0, 0, 1, 0], [batchSize, height, width - 1, numOutputPlanes]), tf.zeros([batchSize, height, 1, numOutputPlanes]), ], axis = 2)
#         #S_neighbors = tf.stack([S_neighbor_up, S_neighbor_down, S_neighbor_left, S_neighbor_right], axis = 4)
#         S_neighbors = (S_neighbor_up + S_neighbor_down + S_neighbor_left + S_neighbor_right) / 4
#         DS = tf.reduce_sum(tf.multiply(D_diff, tf.expand_dims(S_neighbors, 3)), axis=4)
#         #test = tf.multiply(D_diff, tf.expand_dims(S_neighbors, 3))
#         #S_diff = tf.tile(tf.reduce_sum(S_neighbors, axis=3, keep_dims=True), [1, 1, 1, numOutputPlanes]) - S_neighbors
#         S_diff = tf.ones(S_neighbors.shape) - S_neighbors
#         pass
#     P = tf.clip_by_value(S, 1e-4, 1)
#     DS = tf.clip_by_value(DS / 0.5, 1e-4, 1)
#     S_diff = tf.clip_by_value(S_diff, 1e-4, 1)
#     #return tf.nn.softmax(-beta * (-coef[0] * tf.log(P) + coef[1] * tf.log(DS) + coef[2] * tf.log(S_diff))), tf.nn.softmax(tf.log(P)), 1 - tf.clip_by_value(DS / 2, 0, 1), 1 - S_diff, 1 - tf.clip_by_value(tf.multiply(D_diff, tf.expand_dims(S_neighbors, 3)) / 2, 0, 1), S_neighbors, D_diff
#     return tf.nn.softmax(-beta * (-coef[0] * tf.log(P) + coef[1] * tf.log(DS) + coef[2] * tf.log(S_diff)))

def planeDepthsModule(plane_parameters, width, height, info):
    urange = (tf.range(width, dtype=tf.float32) / (width + 1) * (info[16] + 1) - info[2]) / info[0]
    urange = tf.tile(tf.reshape(urange, [1, -1]), [height, 1])
    vrange = (tf.range(height, dtype=tf.float32) / (height + 1) * (info[17] + 1) - info[6]) / info[5]
    vrange = tf.tile(tf.reshape(vrange, [-1, 1]), [1, width])
            
    ranges = tf.stack([urange, np.ones([height, width]), -vrange], axis=2)
    ranges = tf.reshape(ranges, [-1, 3])
            
    planesD = tf.norm(plane_parameters, axis=1, keep_dims=True)
    planesD = tf.clip_by_value(planesD, 1e-5, 10)
    planesNormal = tf.div(tf.negative(plane_parameters), tf.tile(planesD, [1, 3]))

    normalXYZ = tf.matmul(ranges, planesNormal, transpose_b=True)
    normalXYZ = tf.multiply(tf.sign(normalXYZ), tf.clip_by_value(tf.abs(normalXYZ), 1e-4, 1000000))
    normalXYZ = tf.reciprocal(normalXYZ)
    plane_depths = tf.negative(normalXYZ) * tf.reshape(planesD, [-1])
    plane_depths = tf.reshape(plane_depths, [height, width, -1])

    plane_depths = tf.clip_by_value(plane_depths, 0, 10)
    
    return plane_depths

def planeNormalsModule(plane_parameters, width, height):
    planesD = tf.norm(plane_parameters, axis=-1, keep_dims=True)
    planesD = tf.clip_by_value(planesD, 1e-4, 10)
    planesNormal = tf.div(tf.negative(plane_parameters), planesD)

    #plane_normals = tf.tile(tf.reshape(planesNormal, [1, 1, -1, 3]), [height, width, 1, 1])
    #plane_normals = tf.reshape(planesNormal, [1, 1, -1, 3])
    return planesNormal

def gaussian(k=5, sig=0):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    if sig == 0:
        sig = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        pass
    ax = np.arange(-k // 2 + 1., k // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)

def meanfieldModuleLayer(layerSegmentations, planeDepths, numOutputPlanes = 20, numLayers=2, coef = [1, 1, 1], beta = 1, iteration = 0, sigmaDepthDiff = 0.5):
    batchSize = int(planeSegmentations.shape[0])
    height = int(planeSegmentations.shape[1])
    width = int(planeSegmentations.shape[2])

    minDepthDiff = 0.1
    #P = planeSegmentations
    #S = tf.one_hot(tf.argmax(planeSegmentations, 3), depth=numOutputPlanes)
    kernel_size = 9
    neighbor_kernel_array = gaussian(kernel_size)
    neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
    neighbor_kernel_array /= neighbor_kernel_array.sum()
    neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
    neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])


    layerDepths = []
    layerSs = []
    for layer in xrange(numLayers):
        S = tf.one_hot(tf.argmax(planeSegmentations, 3), depth=numOutputPlanes)
        layerDepth = tf.reduce_sum(planeDepths * S, 3, keep_dims=True)
        layerSs.append(S)
        layerDepths.append(layerDepth)

    DSs = []
    conflictDs = []
    conflictDepthThreshold = 0.1
    
    for layer in xrange(numLayers):        
        DS_diff = tf.exp(-tf.pow(1 - tf.clip_by_value(tf.abs(planeDepths - layerDepths[layer]), 0, 1), 2) / sigmaDepthDiff) - tf.exp(-1 / sigmaDepthDiff) * layerSs[layer]
        DS = tf.nn.depthwise_conv2d(DS_diff, tf.tile(neighbor_kernel, [1, 1, numOutputPlanes, 1]), strides=[1, 1, 1, 1], padding='SAME')
        DSs.append(DS)
        
        conflictD = tf.zeros((batchSize, height, width, 1))
        if layer > 0:
            minDepth = tf.min(tf.concat(layerDepths[:layer - 1], axis=3), axis=3, keep_dims=True)
            conflictD = tf.maximum(conflictD, layerDepths[layer] - minDepth)
            pass
        if layer < numLayers - 1:
            maxDepth = tf.max(tf.concat(layerDepths[layer + 1:], axis=3), axis=3, keep_dims=True)
            conflictD = tf.maximum(conflictD, maxDepth -  layerDepths[layer])
            pass
        conflictDs.append(tf.cast(conflictD > conflictDepthThreshold, tf.float32))

        
    P = tf.clip_by_value(P, 1e-4, 1)
    confidence = P * tf.exp(-coef[1] * DS)
    refined_segmentation = tf.nn.softmax(tf.log(confidence))
    return refined_segmentation

def calcImageDiff(images, kernel_size = 9):
    neighbor_kernel_array = gaussian(kernel_size)
    neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
    neighbor_kernel_array /= neighbor_kernel_array.sum()
    neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = -1
    neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
    neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])

    image_diff = tf.nn.depthwise_conv2d(images, tf.tile(neighbor_kernel, [1, 1, 3, 1]), strides=[1, 1, 1, 1], padding='SAME')
    image_diff = tf.pow(image_diff, 2)
    image_diff = tf.reduce_sum(image_diff, axis=3, keep_dims=True)
    var_image_diff =  tf.reduce_mean(image_diff, axis=[1, 2, 3], keep_dims=True)
    #image_diff = image_diff
    #image_diff = tf.exp(-image_diff)
    #image_diff = tf.nn.max_pool(image_diff, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    return image_diff, var_image_diff
    
def meanfieldModule(planeSegmentations, planeDepths, planesY, imageDiff, numOutputPlanes = 20, coef = [1, 1, 1], beta = 1, iteration = 0, maxDepthDiff = 0.2, varDepthDiff = 0.5, kernel_size = 9):
    batchSize = int(planeSegmentations.shape[0])
    height = int(planeSegmentations.shape[1])
    width = int(planeSegmentations.shape[2])

    P = planeSegmentations


    #minDepthDiff = 0.1
    #normalDotThreshold = np.cos(np.deg2rad(30))
    #N_diff = tf.matmul(planeNormals, planeNormals, transpose_b=True)
    #N_diff_mask = tf.cast((N_diff < normalDotThreshold), tf.float) + tf.diag(tf.ones(numOutputPlanes))
    #N_diff = tf.clip(N_diff, minDepthDiff, 1)
    #N_diff_mask = tf.expand_dims(tf.expand_dims(N_diff_mask, 1), 1)

    #D_diff = (D_diff - minDepthDiff) * N_diff_mask + minDepthDiff


    #confidenceThreshold = 0.00
    #P_truncated = P * (P >= confidenceThreshold).astype(tf.float)
    S = tf.one_hot(tf.argmax(planeSegmentations, 3), depth=numOutputPlanes)

    # D = tf.tile(tf.expand_dims(planeDepths, -1), [1, 1, 1, 1, numOutputPlanes])
    # D_transpose = tf.tile(tf.expand_dims(planeDepths, 3), [1, 1, 1, numOutputPlanes, 1])
    # D_diff = tf.abs(D - D_transpose)
    # DS_weight = tf.exp(-tf.pow(tf.clip_by_value(1 - D_diff / maxDepthDiff, 0, 1), 2) / sigmaDepthDiff)
    # DS_diff = tf.reduce_sum(DS_weight * tf.expand_dims(S, 3), axis=4) - tf.exp(-1 / sigmaDepthDiff) * S

    
    
    
    depthWeight = 50.0
    colorWeight = 50.0
    normalY = tf.reduce_sum(S * tf.reshape(planesY, [-1, 1, 1, numOutputPlanes]), axis=3, keep_dims=True)
    depth_diff = (planeDepths - tf.reduce_sum(planeDepths * S, 3, keep_dims=True)) * normalY
    depth_diff = tf.concat([depth_diff[:, :, :, :numOutputPlanes - 1], (1 - S[:, :, :, numOutputPlanes - 1:numOutputPlanes])], axis=3)
    DS_diff = (1 - tf.exp(-tf.pow(tf.minimum(depth_diff, maxDepthDiff), 2) / varDepthDiff)) + (1 - S) * (1 / depthWeight + (colorWeight / depthWeight) * imageDiff)


    #DS_diff = tf.exp(-tf.pow(1 - tf.clip_by_value(tf.abs(planeDepths - tf.reduce_sum(planeDepths * S, 3, keep_dims=True)), 0, 1), 2) / 0.5) - tf.exp(-1 / 0.5) * S

    neighbor_kernel_array = gaussian(kernel_size)
    neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
    neighbor_kernel_array /= neighbor_kernel_array.sum()
    neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
    neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])
    
    DS = tf.nn.depthwise_conv2d(DS_diff, tf.tile(neighbor_kernel, [1, 1, numOutputPlanes, 1]), strides=[1, 1, 1, 1], padding='SAME')
    

    P = tf.clip_by_value(P, 1e-4, 1)
    confidence = P * tf.exp(-coef[1] * DS)
    #confidence = coef[0] * P + tf.exp(-coef[1] * DS) + tf.exp(-coef[2] * S_diff)
    #confidence[:, :, :, numOutputPlanes] = 1e-4
    #confidence = tf.clip(confidence, 1e-4, 1)
    refined_segmentation = tf.nn.softmax(tf.log(confidence))
    return refined_segmentation, {'diff': DS}


def segmentationRefinementModule(planeSegmentations, planeDepths, planesY, imageDiff, numOutputPlanes = 20, numIterations=20, kernel_size = 9):

    # kernel_size = 9
    # neighbor_kernel_array = gaussian(kernel_size)
    # neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
    # neighbor_kernel_array /= neighbor_kernel_array.sum()
    # neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = -1
    # neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
    # neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])

    #maxDepthDiff = tf.Variable(0.3)
    #sigmaDepthDiff = tf.Variable(0.5)
    maxDepthDiff = 0.2
    varDepthDiff = pow(0.2, 2)
    
    
    refined_segmentation = planeSegmentations
    for _ in xrange(numIterations):
        refined_segmentation, _ = meanfieldModule(refined_segmentation, planeDepths, planesY, imageDiff, numOutputPlanes=numOutputPlanes, maxDepthDiff=maxDepthDiff, varDepthDiff=varDepthDiff, kernel_size = kernel_size)
        continue
    return refined_segmentation, {}


def meanfieldModuleBoundary(planeSegmentations, originalSegmentations, planeDepths, occlusionBoundary = 0, smoothBoundary = 0, numOutputPlanes = 20, coef = [1, 10, 1], beta = 1, iteration = 0, sigmaDepthDiff = 0.5):
    batchSize = int(planeSegmentations.shape[0])
    height = int(planeSegmentations.shape[1])
    width = int(planeSegmentations.shape[2])

    #S = tf.one_hot(tf.argmax(planeSegmentations, 3), depth=numOutputPlanes)
    #D_diff = tf.clip_by_value(tf.abs(planeDepths - tf.reduce_sum(planeDepths * S, 3, keep_dims=True)), 0, 1) * smoothBoundary + tf.clip_by_value(1 - smoothBoundary - occlusionBoundary, 0, 1)
    #DS_diff = tf.exp(-tf.pow(1 - D_diff, 2) / sigmaDepthDiff) - tf.exp(-1 / sigmaDepthDiff) * S
    #DS_diff = DS_diff * smoothBoundary + (tf.exp(-1 / sigmaDepthDiff) * occlusionBoundary + tf.clip_by_value(1 - smoothBoundary - occlusionBoundary, 0, 1)) * (1 - S)

    maxDepthDiff = 0.5
    S = planeSegmentations
    D_diff = tf.abs(planeDepths - tf.reduce_sum(planeDepths * S, 3, keep_dims=True))
    DS_diff = tf.clip_by_value(D_diff / maxDepthDiff, 0, 1)
    DS_diff = DS_diff * (1 - occlusionBoundary)
    #+ (1 - S) * occlusionBoundary * 0.1
    
    kernel_size = 5
    neighbor_kernel_array = gaussian(kernel_size)
    neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
    neighbor_kernel_array /= neighbor_kernel_array.sum()
    neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
    neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])

    DS = tf.nn.depthwise_conv2d(DS_diff, tf.tile(neighbor_kernel, [1, 1, numOutputPlanes, 1]), strides=[1, 1, 1, 1], padding='VALID')
    padding = (kernel_size - 1) / 2
    DS = tf.pad(DS, paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]])
    
    P = originalSegmentations
    P = tf.clip_by_value(P, 1e-4, 1)
    confidence = P * tf.exp(-coef[1] * DS)
    #confidence = coef[0] * P + tf.exp(-coef[1] * DS) + tf.exp(-coef[2] * S_diff)
    #confidence[:, :, :, numOutputPlanes] = 1e-4
    #confidence = tf.clip(confidence, 1e-4, 1)
    refined_segmentation = tf.nn.softmax(tf.log(confidence))
    return refined_segmentation


def segmentationRefinementModuleBoundary(planeSegmentations, planeDepths, occlusionBoundary = 0, smoothBoundary = 0, numOutputPlanes = 20, numIterations=20):
    #maxDepthDiff = tf.Variable(0.3)
    #sigmaDepthDiff = tf.Variable(0.5)
    maxDepthDiff = 0.3
    sigmaDepthDiff = 0.5

    refined_segmentation = planeSegmentations

    #occlusionBoundary = tf.slice(boundaries, [0, 0, 0, 1], [boundaries.shape[0], boundaries.shape[1], boundaries.shape[2], 1])
    #smoothBoundary = tf.slice(boundaries, [0, 0, 0, 2], [boundaries.shape[0], boundaries.shape[1], boundaries.shape[2], 1])
    for _ in xrange(numIterations):
        refined_segmentation = meanfieldModuleBoundary(refined_segmentation, planeSegmentations, planeDepths, occlusionBoundary=occlusionBoundary, smoothBoundary=smoothBoundary, numOutputPlanes=numOutputPlanes, sigmaDepthDiff=sigmaDepthDiff)
        continue
    return refined_segmentation


def planeMapModule(depth, normal, ranges):
    #ranges = tf.reshape(ranges, [-1, 3])

    planes = tf.reduce_sum(normal * ranges, 3, keep_dims=True) * depth * normal
    return planes
    
# def planeFittingModule(depth, normal, numPlanes=50, numGlobalPlanes=20, planeAreaThreshold=3*4):
#     width = int(depth.shape[2])
#     height = int(depth.shape[1])

#     focalLength = 517.97
#     urange = (tf.range(width, dtype=tf.float32) / (width + 1) - 0.5) / focalLength * 641
#     urange = tf.tile(tf.reshape(urange, [1, -1]), [height, 1])
#     vrange = (tf.range(height, dtype=tf.float32) / (height + 1) - 0.5) / focalLength * 481
#     vrange = tf.tile(tf.reshape(vrange, [-1, 1]), [1, width])
            
#     ranges = tf.stack([urange, tf.ones([height, width]), -vrange], axis=2)
#     ranges = tf.expand_dims(ranges, 0)

#     batchSize = int(depth.shape[0])
#     planeDiffThreshold = 0.1
#     #plane parameter for each pixel
#     planeMap = planeMapModule(depth, normal, ranges)
    
#     kernel_size = 3
#     neighbor_kernel_array = gaussian(kernel_size)
#     neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
#     neighbor_kernel_array /= neighbor_kernel_array.sum()
#     neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
#     neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])
#     #smoothedPlaneMap = tf.nn.depthwise_conv2d(planeMap, tf.tile(neighbor_kernel, [1, 1, 3, 1]), strides=[1, 1, 1, 1], padding='SAME')
#     median_kernel_array = np.zeros((3, 3, 1, 9))
#     for index in xrange(9):
#         median_kernel_array[index / 3, index % 3, 0, index] = 1
#         continue
#     median_kernel = tf.constant(median_kernel_array.reshape(-1), shape=median_kernel_array.shape, dtype=tf.float32)
#     smoothedPlaneMap = tf.nn.depthwise_conv2d(planeMap, tf.tile(median_kernel, [1, 1, 3, 1]), strides=[1, 1, 1, 1], padding='SAME')
#     smoothedPlaneMap, _ = tf.nn.top_k(tf.reshape(smoothedPlaneMap, [batchSize, height, width, 3, 9]), k=5)
#     planeMap = tf.squeeze(tf.slice(smoothedPlaneMap, [0, 0, 0, 0, 4], [batchSize, height, width, 3, 1]), axis=4)

#     #planeDiff = tf.norm(planeMap - tf.nn.depthwise_conv2d(planeMap, tf.tile(neighbor_kernel, [1, 1, 3, 1]), strides=[1, 1, 1, 1], padding='SAME'), axis=3, keep_dims=True)
#     smoothedPlaneMap = tf.nn.depthwise_conv2d(planeMap, tf.tile(median_kernel, [1, 1, 3, 1]), strides=[1, 1, 1, 1], padding='SAME')
#     planeDiff = tf.reduce_max(tf.norm(tf.expand_dims(planeMap, -1) - tf.reshape(smoothedPlaneMap, [batchSize, height, width, 3, 9]), axis=3, keep_dims=True), axis=4)
#     boundaryMask = tf.cast(tf.less(planeDiff, planeDiffThreshold), tf.float32)
    
#     #opening
#     erosionKernel = np.array([[-1, 0, -1], [0, 0, 0], [-1, 0, -1]], dtype=np.float32).reshape([3, 3, 1])
#     dilationKernel = np.array([[-1, 0, -1], [0, 0, 0], [-1, 0, -1]], dtype=np.float32).reshape([3, 3, 1])
#     boundaryMask = tf.nn.erosion2d(boundaryMask, kernel=erosionKernel, strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    
#     #region indices
#     assignment = tf.reshape(tf.range(batchSize * height * width, dtype=tf.float32) + 1, [batchSize, height, width, 1]) * boundaryMask
#     with tf.variable_scope("flooding") as scope:
#         scope.reuse_variables()
#         for _ in xrange(width / 2):
#             assignment = tf.nn.max_pool(assignment, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool') * boundaryMask
#             continue
#         pass
#     #inds, mask, count = tf.unique_with_counts(tf.concat([tf.constant(0, shape=[1], dtype=tf.float32), tf.reshape(assignment, [-1])], axis=0))
#     #ignoredInds = tf.range(count.shape, dtype=tf.float32) * tf.less(count, planeAreaThreshold)
#     assignment = tf.reshape(assignment, [-1])
    
#     #find unique regions
#     inds, mask, count = tf.unique_with_counts(assignment)
#     ignoredInds = tf.boolean_mask(inds, tf.less(count, planeAreaThreshold))
#     assignment = assignment * (1 - tf.reduce_max(tf.cast(tf.equal(tf.expand_dims(assignment, -1), tf.expand_dims(ignoredInds, 0)), tf.float32), axis=1))
#     inds, mask, count = tf.unique_with_counts(tf.concat([tf.constant(0, shape=[1], dtype=tf.float32), assignment], axis=0))
        
#     mask = tf.slice(mask, [1], [batchSize * height * width])
#     mask = tf.reshape(mask, [batchSize, height, width, 1])
#     #inds = tf.boolean_mask(inds, tf.greater(count, width * height / (16 * 16)))
#     batchInds = tf.equal(tf.cast(tf.tile(tf.reshape(inds - 1, [1, -1]), [batchSize, 1]), tf.int32) / (width * height), tf.expand_dims(tf.range(batchSize), -1))
#     counts = tf.count_nonzero(batchInds, axis=1)
#     counts = tf.concat([tf.constant([1], dtype=tf.int64), counts], axis=0)
#     counts = tf.slice(tf.cumsum(counts), [0], [batchSize])
#     batchPlaneInds = tf.reshape(tf.range(numPlanes), [1, -1]) + tf.cast(tf.reshape(counts, [-1, 1]), tf.int32)
#     #batchPlaneInds = tf.tile(tf.reshape(tf.range(numPlanes, dtype=tf.int32) + 1, [1, 1, 1, -1]), [batchSize, 1, 1, 1])
#     planeMasks = tf.cast(tf.equal(mask, tf.reshape(batchPlaneInds, [batchSize, 1, 1, numPlanes])), tf.float32)

#     planeMasks_test = planeMasks


#     planeAreas = tf.clip_by_value(tf.reduce_sum(planeMasks, axis=[1, 2]), 1e-4, width * height)
#     #planeAreas, sortInds = tf.nn.top_k(planeAreas, k=numPlanes)
#     #sortMap = tf.one_hot(sortInds, depth=numPlanes, axis=1)
#     #planeMasks = tf.reshape(tf.matmul(tf.reshape(planeMasks, [-1, height * width, numPlanes]), sortMap), [-1, height, width, numPlanes])

#     #fit plane based on mask
#     planesNormal = tf.reduce_sum(tf.expand_dims(normal, 3) * tf.expand_dims(planeMasks, -1), axis=[1, 2]) / tf.expand_dims(planeAreas, -1)
#     planesNormal = tf.nn.l2_normalize(planesNormal, 2)

#     weightedABC = tf.transpose(tf.reshape(tf.matmul(tf.reshape(ranges, [-1, 3]), tf.reshape(planesNormal, [-1, 3]), transpose_b=True), [height, width, batchSize, numPlanes]), [2, 0, 1, 3])
#     planesD = tf.reduce_sum(weightedABC * depth * planeMasks, axis=[1, 2]) / planeAreas
#     planesD = tf.expand_dims(planesD, -1)
#     planes = planesNormal * planesD
    
#     #globalPlanes = tf.slice(planes, [0, 0, 0], [batchSize, numGlobalPlanes, 3])
#     #planes = tf.transpose(tf.matmul(planes, sortMap, transpose_a=True), [0, 2, 1])
#     #planesNormal = tf.slice(planesNormal, [0, 0, 0], [batchSize, numGlobalPlanes, 3])
#     #planesD = tf.slice(planesD, [0, 0, 0], [batchSize, numGlobalPlanes, 1])

#     normalDotThreshold = np.cos(np.deg2rad(5))
#     distanceThreshold = 0.05
#     X = depth * tf.expand_dims(urange, -1)
#     Y = depth
#     Z = -depth * tf.expand_dims(vrange, -1)
#     XYZ = tf.concat([X, Y, Z], axis=3)
#     XYZ = tf.reshape(XYZ, [-1, height * width, 3])
    
#     planesNormal = -planesNormal
#     distance = tf.reshape(tf.abs(tf.matmul(XYZ, planesNormal, transpose_b=True) + tf.reshape(planesD, [-1, 1, numPlanes])), [-1, height, width, numPlanes])
#     angle = tf.reshape(tf.abs(tf.matmul(tf.reshape(normal, [-1, height * width, 3]), planesNormal, transpose_b=True)), [-1, height, width, numPlanes])

#     explainedPlaneMasks = tf.cast(tf.logical_and(tf.greater(angle, normalDotThreshold), tf.less(distance, distanceThreshold)), tf.float32)
#     explainedPlaneMasks = tf.nn.dilation2d(explainedPlaneMasks, filter=np.tile(dilationKernel, [1, 1, numPlanes]), strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
#     explainedPlaneMasks = tf.nn.erosion2d(explainedPlaneMasks, kernel=np.tile(erosionKernel, [1, 1, numPlanes]), strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')    

#     with tf.variable_scope("expansion") as scope:
#         scope.reuse_variables()
#         for _ in xrange(width / 6):
#             planeMasks = tf.nn.max_pool(planeMasks, ksize=[1, 13, 13, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool') * explainedPlaneMasks
#             continue
#         pass
        
#     planeAreas = tf.reduce_sum(planeMasks, axis=[1, 2])
#     planeAreas, sortInds = tf.nn.top_k(planeAreas, k=numPlanes)
#     sortMap = tf.one_hot(sortInds, depth=numPlanes, axis=1)
#     planeMasks = tf.reshape(tf.matmul(tf.reshape(planeMasks, [-1, height * width, numPlanes]), sortMap), [-1, height, width, numPlanes])
#     planes = tf.transpose(tf.matmul(planes, sortMap, transpose_a=True), [0, 2, 1])

#     #remove duplicate planes by expanding each plane mask, if two masks coincide, remove one of them
#     substractionMatrix = -tf.cast(tf.less(tf.reshape(tf.range(numPlanes), [-1, 1]), tf.reshape(tf.range(numPlanes), [1, -1])), tf.float32) + tf.eye(numPlanes)
#     substractionMatrix = tf.tile(tf.expand_dims(substractionMatrix, 0), [batchSize, 1, 1])
#     planeMasks = tf.clip_by_value(tf.reshape(tf.matmul(tf.reshape(planeMasks, [-1, height * width, numPlanes]), substractionMatrix), [-1, height, width, numPlanes]), 0, 1)


#     planeMasksWithoutBoundary = planeMasks * boundaryMask
#     planeAreas = tf.reduce_sum(planeMasksWithoutBoundary, axis=[1, 2])
#     maxMeanDepthThreshold = 10
#     planeAreas = tf.clip_by_value(planeAreas, 1e-4, width * height)
#     #validPlaneMask = tf.cast(tf.logical_and(tf.logical_or(tf.greater(planeAreas, planeAreaThreshold), tf.equal(tf.argmax(np.abs(planes), 2), 2)), tf.less(tf.reduce_sum(planeMasksWithoutBoundary * depth, axis=[1, 2]) / planeAreas, maxMeanDepthThreshold)), tf.float32)
#     #validPlaneMask = tf.cast(tf.equal(tf.argmax(np.abs(planes), 2), 2), tf.float32)
#     #planeAreas, sortInds = tf.nn.top_k(planeAreas, k=numPlanes)
#     #sortMap = tf.one_hot(sortInds, depth=numPlanes, axis=1)
#     #planeMasks = tf.reshape(tf.matmul(tf.reshape(planeMasks, [-1, height * width, numPlanes]), sortMap), [-1, height, width, numPlanes])

#     if False:
#         planes = tf.transpose(tf.matmul(planes, sortMap, transpose_a=True), [0, 2, 1])
#     else:
#         #fit planes based on merged masks
#         planesNormal = tf.reduce_sum(tf.expand_dims(normal, 3) * tf.expand_dims(planeMasksWithoutBoundary, -1), axis=[1, 2]) / tf.expand_dims(planeAreas, -1)
#         planesNormal = tf.nn.l2_normalize(planesNormal, 2)

#         weightedABC = tf.transpose(tf.reshape(tf.matmul(tf.reshape(ranges, [-1, 3]), tf.reshape(planesNormal, [-1, 3]), transpose_b=True), [height, width, batchSize, numPlanes]), [2, 0, 1, 3])
#         planesD = tf.reduce_sum(weightedABC * depth * planeMasksWithoutBoundary, axis=[1, 2]) / planeAreas
#         planesD = tf.expand_dims(planesD, -1)
#         planes = planesNormal * planesD
#         pass

#     validPlaneMask = tf.cast(tf.less(tf.reduce_sum(planeMasksWithoutBoundary * depth, axis=[1, 2]) / planeAreas, maxMeanDepthThreshold), tf.float32)
#     planeMasks = planeMasks * tf.expand_dims(tf.expand_dims(validPlaneMask, 1), 1)
#     planes = planes * tf.expand_dims(validPlaneMask, -1)
#     planeAreas = planeAreas * validPlaneMask
            

#     # planeAreas = tf.reduce_sum(localPlaneMasks, axis=[1, 2])
#     # planeAreas, sortInds = tf.nn.top_k(planeAreas, k=numPlanes)
#     # sortMap = tf.one_hot(sortInds, depth=numPlanes, axis=1)
#     # localPlaneMasks = tf.reshape(tf.matmul(tf.reshape(localPlaneMasks, [-1, height * width, numPlanes]), sortMap), [-1, height, width, numPlanes])
#     # localPlanes = tf.transpose(tf.matmul(localPlanes, sortMap, transpose_a=True), [0, 2, 1])

#     # substractionMatrix = -tf.cast(tf.less(tf.reshape(tf.range(numPlanes), [-1, 1]), tf.reshape(tf.range(numPlanes), [1, -1])), tf.float32) + tf.eye(numPlanes)
#     # substractionMatrix = tf.tile(tf.expand_dims(substractionMatrix, 0), [batchSize, 1, 1])
#     # localPlaneMasks = tf.clip_by_value(tf.reshape(tf.matmul(tf.reshape(localPlaneMasks, [-1, height * width, numPlanes]), substractionMatrix), [-1, height, width, numPlanes]), 0, 1)


#     # planeMasksWithoutBoundary = localPlaneMasks * boundaryMask
#     # planeAreas = tf.reduce_sum(planeMasksWithoutBoundary, axis=[1, 2])
#     # maxMeanDepthThreshold = 10
#     # #validPlaneMask = tf.cast(tf.logical_and(tf.logical_or(tf.greater(planeAreas, planeAreaThreshold), tf.equal(tf.argmax(np.abs(planes), 2), 2)), tf.less(tf.reduce_sum(planeMasksWithoutBoundary * depth, axis=[1, 2]) / planeAreas, maxMeanDepthThreshold)), tf.float32)
#     # #validPlaneMask = tf.cast(tf.equal(tf.argmax(np.abs(planes), 2), 2), tf.float32)
#     # validPlaneMask = tf.cast(tf.less(tf.reduce_sum(planeMasksWithoutBoundary * depth, axis=[1, 2]) / planeAreas, maxMeanDepthThreshold), tf.float32)
#     # localPlanes = localPlanes * tf.expand_dims(validPlaneMask, -1)
#     # localPlaneMasks = localPlaneMasks * tf.expand_dims(tf.expand_dims(validPlaneMask, 1), 1)
#     # planeAreas = planeAreas * validPlaneMask
#     # planeAreas, sortInds = tf.nn.top_k(planeAreas, k=numPlanes)
#     # sortMap = tf.one_hot(sortInds, depth=numPlanes, axis=1)
#     # localPlaneMasks = tf.reshape(tf.matmul(tf.reshape(localPlaneMasks, [-1, height * width, numPlanes]), sortMap), [-1, height, width, numPlanes])

#     # planeMasksWithoutBoundary = localPlaneMasks * boundaryMask
#     # planeAreas = tf.clip_by_value(planeAreas, 1e-4, width * height)
#     # planesNormal = tf.reduce_sum(tf.expand_dims(normal, 3) * tf.expand_dims(planeMasksWithoutBoundary, -1), axis=[1, 2]) / tf.expand_dims(planeAreas, -1)
#     # planesNormal = tf.nn.l2_normalize(planesNormal, 2)

#     # weightedABC = tf.transpose(tf.reshape(tf.matmul(tf.reshape(ranges, [-1, 3]), tf.reshape(planesNormal, [-1, 3]), transpose_b=True), [height, width, batchSize, numPlanes]), [2, 0, 1, 3])
#     # planesD = tf.reduce_sum(weightedABC * depth * planeMasksWithoutBoundary, axis=[1, 2]) / planeAreas
#     # planesD = tf.expand_dims(planesD, -1)
#     # localPlanes = planesNormal * planesD
    

#     #find local ground truth
#     urange = tf.reshape(tf.range(width, dtype=tf.float32), [1, -1, 1])
#     planeXs = tf.reduce_max(planeMasks, axis=1)
#     planeMinX = width - tf.reduce_max(planeXs * (float(width) - urange), axis=1)
#     planeMaxX = tf.reduce_max(planeXs * urange, axis=1)

#     vrange = tf.reshape(tf.range(height, dtype=tf.float32), [1, -1, 1])
#     planeYs = tf.reduce_max(planeMasks, axis=2)
#     planeMinY = height - tf.reduce_max(planeYs * (float(height) - vrange), axis=1)
#     planeMaxY = tf.reduce_max(planeYs * vrange, axis=1)

#     planeBoxes = tf.stack([planeMinX, planeMaxX, planeMinY, planeMaxY], axis=2)

#     localPlaneWidthThreshold = 64
#     localPlaneHeightThreshold = 64
#     globalPlaneAreaThreshold = 16 * 16
#     globalPlaneWidthThreshold = 8
    
#     globalPlaneMask = tf.logical_or(tf.greater(planeMaxX - planeMinX, localPlaneWidthThreshold), tf.greater(planeMaxY - planeMinY, localPlaneHeightThreshold))
#     globalPlaneMask = tf.logical_and(globalPlaneMask, tf.greater((planeMaxX - planeMinX) * (planeMaxY - planeMinY), globalPlaneAreaThreshold))
#     globalPlaneMask = tf.logical_and(globalPlaneMask, tf.greater(planeAreas / (planeMaxY + 1 - planeMinY), globalPlaneWidthThreshold))
#     #globalPlaneMask = tf.cast(tf.squeeze(globalPlaneMask, axis=[2]), tf.float32)
#     globalPlaneMask = tf.cast(globalPlaneMask, tf.float32)
#     weightedPlaneAreas = globalPlaneMask * (planeAreas + height * width) + (1 - globalPlaneMask) * planeAreas
#     planeAreas, sortInds = tf.nn.top_k(weightedPlaneAreas, k=numPlanes)
#     sortMap = tf.one_hot(sortInds, depth=numPlanes, axis=1)
#     planeMasks = tf.reshape(tf.matmul(tf.reshape(planeMasks, [-1, height * width, numPlanes]), sortMap), [-1, height, width, numPlanes])
#     planes = tf.transpose(tf.matmul(planes, sortMap, transpose_a=True), [0, 2, 1])
#     planeBoxes = tf.transpose(tf.matmul(planeBoxes, sortMap, transpose_a=True), [0, 2, 1])
#     globalPlaneMask = tf.squeeze(tf.matmul(tf.expand_dims(globalPlaneMask, 1), sortMap), axis=1)
    


#     #boundary ground truth
#     boundary = tf.nn.max_pool(planeMasks, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
    
#     boundaryType = tf.cast(tf.round(tf.reduce_sum(boundary, axis=3, keep_dims=True)), tf.int32)
#     singleBoundary = tf.cast(tf.equal(tf.reduce_sum(boundary - planeMasks, axis=3, keep_dims=True), 1), tf.float32)

#     commonBoundary = tf.cast(tf.equal(boundaryType, 2), tf.float32)
#     #boundary = boundary * commonBoundary
#     boundaryCoef = tf.cast(tf.round(tf.cumsum(boundary, axis=3)), tf.float32)

#     #boundary_1 = tf.cast(tf.equal(boundaryCoef, 1), tf.float32) * boundary
#     #boundary_1 = tf.cast(tf.equal(boundaryCoef, 1), tf.float32) * boundary
    
#     boundaryPlane_1 = tf.reshape(tf.matmul(tf.reshape(tf.cast(tf.equal(boundaryCoef, 1), tf.float32) * boundary, [batchSize, height * width, numPlanes]), planes), [batchSize, height, width, 3])
#     boundaryD_1 = tf.maximum(tf.norm(boundaryPlane_1, axis=3, keep_dims=True), 1e-4)
#     boundaryNormal_1 = boundaryPlane_1 / boundaryD_1
#     boundaryDepth_1 = boundaryD_1 / tf.maximum(tf.reduce_sum(boundaryNormal_1 * ranges, axis=3, keep_dims=True), 1e-4)

#     boundaryPlane_2 = tf.reshape(tf.matmul(tf.reshape(tf.cast(tf.equal(boundaryCoef, 2), tf.float32) * boundary, [batchSize, height * width, numPlanes]), planes), [batchSize, height, width, 3])
#     boundaryD_2 = tf.maximum(tf.norm(boundaryPlane_2, axis=3, keep_dims=True), 1e-4)
#     boundaryNormal_2 = boundaryPlane_2 / boundaryD_2
#     boundaryDepth_2 = boundaryD_2 / tf.maximum(tf.reduce_sum(boundaryNormal_2 * ranges, axis=3, keep_dims=True), 1e-4)

#     depthDiffThreshold = 0.05
#     #occlusionBoundary = tf.cast(tf.greater(tf.abs(boundaryDepth_1 - boundaryDepth_2), depthDiffThreshold), tf.float32) * commonBoundary
#     largerMask = tf.nn.max_pool(tf.cast(tf.greater_equal(boundaryDepth_1, boundaryDepth_2), tf.float32) * commonBoundary, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
#     smallerMask = tf.nn.max_pool(tf.cast(tf.less_equal(boundaryDepth_1, boundaryDepth_2), tf.float32) * commonBoundary, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
#     smoothBoundary = tf.nn.max_pool(largerMask * smallerMask, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
#     #depthDiff = tf.abs(depth - tf.nn.depthwise_conv2d(depth, neighbor_kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     #occlusionBoundary = tf.cast(tf.greater(depthDiff, depthDiffThreshold), tf.float32) * commonBoundary
    
#     #boundaryConvexity = tf.cast(tf.less(tf.reduce_sum(boundaryNormal_1 * boundaryNormal_2, axis=3, keep_dims=True), 0), tf.float32)
#     #convexBoundary = smoothBoundary * boundaryConvexity
#     #concaveBoundary = smoothBoundary * (1 - boundaryConvexity)

    
#     occlusionBoundary = commonBoundary - smoothBoundary

#     singleBoundary = tf.maximum(singleBoundary - tf.nn.max_pool(commonBoundary, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME'), 0)
#     boundaries = tf.concat([singleBoundary, occlusionBoundary, smoothBoundary], axis=3)
#     #boundaries = tf.concat([tf.maximum(tf.minimum((boundaryDepth_1 - boundaryDepth_2) + 0.5, 1), 0)], axis=3)
#     #boundaries = tf.concat([tf.maximum(tf.minimum(boundaryDepth_1 / 10, 1), 0), tf.maximum(tf.minimum(boundaryDepth_2 / 10, 1), 0), tf.maximum(tf.minimum((boundaryDepth_1 - boundaryDepth_2) + 0.5, 1), 0)], axis=3)
#     boundaries = 1 - tf.nn.max_pool(1 - boundaries, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    
#     if True:
#         coef = tf.pow(tf.constant(2, dtype=tf.float64), tf.range(numPlanes, dtype=tf.float64))
#         planeMask = tf.cast(tf.round(tf.tensordot(tf.cast(planeMasks, tf.float64), coef, axes=[[3], [0]])), tf.int64)
#         #localPlaneMask = tf.cast(tf.round(tf.tensordot(tf.cast(localPlaneMasks, tf.float64), coef, axes=[[3], [0]])), tf.int64)
#         #coef = tf.pow(tf.constant(2, dtype=tf.float64), tf.range(numPlanes, dtype=tf.float64))
#         #planeCroppedMask = tf.cast(tf.round(tf.tensordot(tf.cast(tf.greater(planeCroppedMasks, 0.5), tf.float64), coef, axes=[[3], [0]])), tf.int64)
#         numGlobalPlanes = tf.reduce_sum(globalPlaneMask, axis=1)

#         gridScores, gridPlanes, gridMasks = findLocalPlanes(planes, planeMasks)
#         return planes, planeMask, numGlobalPlanes, boundaries, gridScores, gridPlanes, gridMasks

    
#     maskWidth = 32
#     maskHeight = 32
#     planeCroppedMasks = []
#     for batchIndex in xrange(batchSize):
#         boxes = planeBoxes[batchIndex]
#         masks = tf.transpose(planeMasks[batchIndex], [2, 0, 1])
#         croppedMasks = []
#         for planeIndex in xrange(numPlanes):
#         #for planeIndex in xrange(1):
#             box = boxes[planeIndex]
#             mask = masks[planeIndex]
#             #minX = tf.cond(tf.less(planeIndex, tf.numValidPlanes[batchIndex]), lambda: tf.cast(box[0], tf.int32)
#             minX = tf.cast(box[0], tf.int32)
#             maxX = tf.cast(box[1], tf.int32)
#             minY = tf.cast(box[2], tf.int32)
#             maxY = tf.cast(box[3], tf.int32)
#             minX = tf.minimum(minX, maxX)
#             minY = tf.minimum(minY, maxY)
#             croppedMask = tf.slice(mask, [minY, minX], [maxY - minY + 1, maxX - minX + 1])
#             #croppedMask = tf.slice(mask, [0, 0], [height - 10, width - 10])
#             croppedMask = tf.image.resize_bilinear(tf.expand_dims(tf.expand_dims(croppedMask, -1), 0), [maskHeight, maskWidth])
#             croppedMasks.append(croppedMask)
#             continue
#         planeCroppedMasks.append(tf.squeeze(tf.concat(croppedMasks, axis=3)))
#         continue
#     planeCroppedMasks = tf.stack(planeCroppedMasks, axis=0)   

#     gridMinX = []
#     gridMaxX = []
#     gridMinY = []
#     gridMaxY = []
#     for stride in [8, 16, 32]:
#         boxSize = stride * 2
#         xs = tf.tile(tf.expand_dims(tf.range(width / stride, dtype=tf.float32) * stride + stride / 2, 0), [height / stride, 1])
#         ys = tf.tile(tf.expand_dims(tf.range(height / stride, dtype=tf.float32) * stride + stride / 2, 1), [1, width / stride])
#         gridMinX.append(tf.reshape(xs - boxSize / 2, [1, -1, 1]))
#         gridMaxX.append(tf.reshape(xs + boxSize / 2, [1, -1, 1]))
#         gridMinY.append(tf.reshape(ys - boxSize / 2, [1, -1, 1]))
#         gridMaxY.append(tf.reshape(ys + boxSize / 2, [1, -1, 1]))
#         continue
    
#     gridMinX = tf.tile(tf.concat(gridMinX, axis=1), [batchSize, 1, 1])
#     gridMaxX = tf.tile(tf.concat(gridMaxX, axis=1), [batchSize, 1, 1])
#     gridMinY = tf.tile(tf.concat(gridMinY, axis=1), [batchSize, 1, 1])
#     gridMaxY = tf.tile(tf.concat(gridMaxY, axis=1), [batchSize, 1, 1])

#     planeMinX = tf.matmul(tf.reshape(planeMinX, [batchSize, 1, numPlanes]), sortMap)
#     planeMaxX = tf.matmul(tf.reshape(planeMaxX, [batchSize, 1, numPlanes]), sortMap)
#     planeMinY = tf.matmul(tf.reshape(planeMinY, [batchSize, 1, numPlanes]), sortMap)
#     planeMaxY = tf.matmul(tf.reshape(planeMaxY, [batchSize, 1, numPlanes]), sortMap)

#     intersection = tf.maximum(tf.minimum(gridMaxX, planeMaxX) - tf.maximum(gridMinX, planeMinX) + 1, 0.) * tf.maximum(tf.minimum(gridMaxY, planeMaxY) - tf.maximum(gridMinY, planeMinY) + 1, 0.)
#     union = (gridMaxX - gridMinX + 1) * (gridMaxY - gridMinY + 1) + (planeMaxX - planeMinX + 1) * (planeMaxY - planeMinY + 1) - intersection
#     IOU = intersection / union
#     maxIOUInds = tf.argmax(IOU, axis=1)
#     maxIOU = tf.reduce_max(IOU, axis=1)
#     IOU = IOU * tf.one_hot(tf.argmax(IOU, 1), depth=IOU.shape[1], axis=1)
#     #IOUThreshold = tf.concat([tf.ones((1, (width / 8) * (height / 8), 1)) * 0.2, tf.ones((1, (width / 16) * (height / 16), 1)) * 0.3, tf.ones((1, (width / 32) * (height / 32), 1)) * 0.7], axis=1)
#     #activeGridMask = tf.one_hot(tf.argmax(IOU, 2), depth=IOU.shape[2], axis=2) * tf.cast(tf.greater(IOU, IOUThreshold), tf.float32)
#     activeGridMask = tf.one_hot(tf.argmax(IOU, 2), depth=IOU.shape[2], axis=2) * (1 - tf.expand_dims(globalPlaneMask, 1))
#     gridScores = tf.reduce_sum(activeGridMask, axis=2, keep_dims=True)
#     activeGridMask = tf.expand_dims(activeGridMask, -1)
#     gridPlanes = tf.reduce_sum(activeGridMask * tf.expand_dims(planes, 1), axis=2)
#     gridMasks = tf.reduce_sum(activeGridMask * tf.expand_dims(tf.transpose(tf.reshape(planeCroppedMasks, [batchSize, -1, numPlanes]), [0, 2, 1]), 1), axis=2)

#     activeGridMask = tf.squeeze(activeGridMask, axis=3)
#     #gridBoxes = tf.reduce_sum(activeGridMask * tf.expand_dims(planeBoxes, 1), axis=2)
#     gridPlaneMinX = tf.reduce_sum(activeGridMask * planeMinX, axis=2, keep_dims=True)
#     gridPlaneMaxX = tf.reduce_sum(activeGridMask * planeMaxX, axis=2, keep_dims=True)
#     gridPlaneMinY = tf.reduce_sum(activeGridMask * planeMinY, axis=2, keep_dims=True)
#     gridPlaneMaxY = tf.reduce_sum(activeGridMask * planeMaxY, axis=2, keep_dims=True)
#     gridWidths = gridMaxX - gridMinX
#     gridHeights = gridMaxY - gridMinY

#     gridOffsetX = ((gridPlaneMinX + gridPlaneMaxX) - (gridMinX + gridMaxX)) / 2 / gridWidths
#     gridOffsetY = ((gridPlaneMinY + gridPlaneMaxY) - (gridMinY + gridMaxY)) / 2 / gridHeights
#     gridW = (gridPlaneMaxX - gridPlaneMinX) / gridWidths
#     gridH = (gridPlaneMaxY - gridPlaneMinY) / gridHeights
#     gridBoxes = tf.concat([gridOffsetX, gridOffsetY, gridW, gridH], axis=2)
    
    
#     offset = 0
#     gridScoresArray = []
#     gridPlanesArray = []
#     gridBoxesArray = []
#     gridMasksArray = []
#     for stride in [8, 16, 32]:
#         numGrids = (width / stride) * (height / stride)
#         gridScoresArray.append(tf.reshape(tf.slice(gridScores, [0, offset, 0], [batchSize, numGrids, 1]), [batchSize, height / stride, width / stride, -1]))
#         gridPlanesArray.append(tf.reshape(tf.slice(gridPlanes, [0, offset, 0], [batchSize, numGrids, 3]), [batchSize, height / stride, width / stride, -1]))
#         gridBoxesArray.append(tf.reshape(tf.slice(gridBoxes, [0, offset, 0], [batchSize, numGrids, 4]), [batchSize, height / stride, width / stride, -1]))
#         gridMasksArray.append(tf.reshape(tf.slice(gridMasks, [0, offset, 0], [batchSize, numGrids, maskWidth * maskHeight]), [batchSize, height / stride, width / stride, -1]))
#         offset += numGrids
#         continue

    
#     if True:
#         coef = tf.pow(tf.constant(2, dtype=tf.float64), tf.range(numPlanes, dtype=tf.float64))
#         planeMask = tf.cast(tf.round(tf.tensordot(tf.cast(planeMasks, tf.float64), coef, axes=[[3], [0]])), tf.int64)
#         #localPlaneMask = tf.cast(tf.round(tf.tensordot(tf.cast(localPlaneMasks, tf.float64), coef, axes=[[3], [0]])), tf.int64)
#         #coef = tf.pow(tf.constant(2, dtype=tf.float64), tf.range(numPlanes, dtype=tf.float64))
#         planeCroppedMask = tf.cast(tf.round(tf.tensordot(tf.cast(tf.greater(planeCroppedMasks, 0.5), tf.float64), coef, axes=[[3], [0]])), tf.int64)
#         numGlobalPlanes = tf.reduce_sum(globalPlaneMask, axis=1)

#         return planes, planeMask, numGlobalPlanes, boundaries, planeBoxes, planeCroppedMask, gridScoresArray, gridPlanesArray, gridBoxesArray, gridMasksArray, maxIOU, maxIOUInds
    
#     # coef = tf.pow(tf.constant(0.9, dtype=tf.float64), tf.range(numPlanes, dtype=tf.float64))
#     # coef = tf.tile(tf.reshape(coef, [1, 1, -1]), [batchSize, 1, 1])
#     # coef = tf.matmul(coef, tf.cast(sortMap, tf.float64), transpose_b=True)
#     # #planeMasks = tf.reshape(tf.matmul(tf.reshape(planeMasks, [-1, height * width, numPlanes]), sortMap), [-1, height, width, numPlanes])

#     # assignment = tf.reduce_max(tf.cast(planeMasks, tf.float64) * tf.expand_dims(coef, axis=2), axis=3, keep_dims=True)
#     # inds, mask, count = tf.unique_with_counts(tf.concat([tf.constant(0), tf.reshape(assignment, [-1])]))
#     # mask = tf.reshape(tf.slice(mask, [1], [batchSize * height * width * 1], [batchSize, height, width, 1])

#     # coef = tf.tile(tf.reshape(coef, [1, 1, -1]), [batchSize, 1, 1])
#     # coef = tf.matmul(coef, tf.cast(sortMap, tf.float64), transpose_b=True)
#     # coef = tf.reshape(tf.range(numPlanes)
#     # planeMasks = tf.cast(tf.equal(mask, tf.tile(, [1, 1, 1, numPlanes]), [batchSize, 1, 1, 1])), tf.float32)
    
#     # planeAreas = tf.clip_by_value(tf.reduce_sum(planeMasks, axis=[1, 2]), 1e-4, width * height)
#     # planeAreas, sortInds = tf.nn.top_k(planeAreas, k=numPlanes)
#     # sortMap = tf.one_hot(sortInds, depth=numPlanes, axis=1)
#     # planeMasks = tf.reshape(tf.matmul(tf.reshape(planeMasks, [-1, height * width, numPlanes]), sortMap), [-1, height, width, numPlanes])

#     #planeAreas = tf.clip_by_value(tf.reduce_sum(planeMasks, axis=[1, 2]), 1e-4, width * height)
    
#     # planesNormal = tf.reduce_sum(tf.expand_dims(normal, 3) * tf.expand_dims(planeMasks, -1), axis=[1, 2]) / tf.expand_dims(planeAreas, -1)
#     # planesNormal = tf.nn.l2_normalize(planesNormal, 2)
#     # weightedABC = tf.reshape(tf.matmul(tf.reshape(ranges, [-1, 3]), tf.reshape(planesNormal, [-1, 3]), transpose_b=True), [batchSize, height, width, numPlanes])
#     # planesD = tf.reduce_sum(weightedABC * depth * planeMasks, axis=[1, 2]) / planeAreas
#     # planesD = tf.expand_dims(planesD, -1)
#     # planes = planesNormal * planesD


#     if True:
#         planeMask = tf.cast(tf.round(tf.tensordot(tf.cast(planeMasks, tf.float64), coef, axes=[[3], [0]])), tf.int64)
#         return planes, planeMask, tf.reduce_sum(validPlaneMask, axis=1)

    
#     globalPlanes = tf.slice(planes, [0, 0, 0], [batchSize, numGlobalPlanes, 3])
#     globalPlaneMasks = tf.slice(planeMasks, [0, 0, 0, 0], [batchSize, height, width, numGlobalPlanes])

#     if True:
#         return planes, planeMasks, tf.reduce_sum(validPlaneMask, axis=1), planeMasks_test, boundaryMask
#     #return globalPlanes, globalPlaneMasks, tf.reduce_sum(validPlaneMask, axis=1)
    
#     globalPlaneMask = tf.reduce_max(globalPlaneMasks, axis=3, keep_dims=True)
#     smallPlaneMasks = tf.clip_by_value(tf.slice(planeMasks, [0, 0, 0, numGlobalPlanes], [batchSize, height, width, numPlanes - numGlobalPlanes]) - globalPlaneMask, 0, 1)
#     smallPlaneMasks = tf.nn.dilation2d(smallPlaneMasks, filter=np.tile(dilationKernel, [1, 1, numPlanes - numGlobalPlanes]), strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
#     smallPlaneMasks = tf.concat([globalPlaneMasks, smallPlaneMasks], axis=3)


#     IOUThreshold = 0.9
#     areaThreshold = 0.25

#     blockSize = 16
#     smallPlaneInds = tf.nn.avg_pool(smallPlaneMasks, ksize=[1, blockSize, blockSize, 1], strides=[1, blockSize, blockSize, 1], padding='VALID')
#     smallPlaneAreas = tf.clip_by_value(tf.reduce_sum(smallPlaneInds, axis=[1, 2], keep_dims=True), 1e-4, width * height)    
#     IOU = smallPlaneInds / smallPlaneAreas
#     inds = smallPlaneInds
#     smallPlaneInds = tf.one_hot(tf.argmax(smallPlaneInds, 3), depth=numPlanes) * tf.cast(tf.greater_equal(IOU, IOUThreshold), tf.float32) * tf.cast(tf.greater(smallPlaneInds, areaThreshold), tf.float32)

#     blockSmallPlaneMasks_16 = tf.reshape(tf.space_to_depth(smallPlaneMasks, block_size=blockSize), [batchSize, height / blockSize, width / blockSize, blockSize * blockSize, numPlanes])
#     blockSmallPlanes_16 = tf.reduce_sum(tf.expand_dims(smallPlaneInds, -1) * tf.expand_dims(tf.expand_dims(planes, 1), 1), axis=3)
#     blockSmallPlaneMasks_16 = tf.reduce_sum(tf.expand_dims(smallPlaneInds, 3) * blockSmallPlaneMasks_16, axis=4)
#     blockPlaneIndicators_16 = tf.reduce_max(smallPlaneInds, axis=3, keep_dims=True)

    
#     blockSize = 32
#     smallPlaneInds = tf.nn.avg_pool(smallPlaneMasks, ksize=[1, blockSize, blockSize, 1], strides=[1, blockSize, blockSize, 1], padding='VALID')
#     smallPlaneAreas = tf.clip_by_value(tf.reduce_sum(smallPlaneInds, axis=[1, 2], keep_dims=True), 1e-4, width * height)    
#     IOU = smallPlaneInds / smallPlaneAreas
#     inds = smallPlaneInds
#     smallPlaneInds = tf.one_hot(tf.argmax(smallPlaneInds, 3), depth=numPlanes) * tf.cast(tf.greater_equal(IOU, IOUThreshold), tf.float32) * tf.cast(tf.greater(smallPlaneInds, areaThreshold), tf.float32)
    
#     blockSmallPlaneMasks_32 = tf.reshape(tf.space_to_depth(smallPlaneMasks, block_size=blockSize), [batchSize, height / blockSize, width / blockSize, blockSize * blockSize, numPlanes])
#     blockSmallPlanes_32 = tf.reduce_sum(tf.expand_dims(smallPlaneInds, -1) * tf.expand_dims(tf.expand_dims(planes, 1), 1), axis=3)
#     blockSmallPlaneMasks_32 = tf.reduce_sum(tf.expand_dims(smallPlaneInds, 3) * blockSmallPlaneMasks_32, axis=4)
#     blockPlaneIndicators_32 = tf.reduce_max(smallPlaneInds, axis=3, keep_dims=True)

#     return globalPlanes, globalPlaneMasks, blockSmallPlanes_16, blockSmallPlaneMasks_16, blockPlaneIndicators_16, blockSmallPlanes_32, blockSmallPlaneMasks_32, blockPlaneIndicators_32, tf.depth_to_space(blockSmallPlaneMasks_16 * blockPlaneIndicators_16, 16), tf.depth_to_space(blockSmallPlaneMasks_32 * blockPlaneIndicators_32, 32), planeMasks_test, planeDiff, boundaryMask


# def planeFittingDepthModule(depth)
#     width = int(depth.shape[2])
#     height = int(depth.shape[1])

#     focalLength = 517.97
#     urange = (tf.range(width, dtype=tf.float32) / (width + 1) - 0.5) / focalLength * 641
#     urange = tf.tile(tf.reshape(urange, [1, -1]), [height, 1])
#     vrange = (tf.range(height, dtype=tf.float32) / (height + 1) - 0.5) / focalLength * 481
#     vrange = tf.tile(tf.reshape(vrange, [-1, 1]), [1, width])
            
#     ranges = tf.stack([urange, tf.ones([height, width]), -vrange], axis=2)
#     ranges = tf.expand_dims(ranges, 0)

#     batchSize = int(depth.shape[0])

#     X = depth * tf.expand_dims(urange, -1)
#     Y = depth
#     Z = -depth * tf.expand_dims(vrange, -1)
#     XYZ = tf.concat([X, Y, Z], axis=3)
#     XYZ = tf.reshape(XYZ, [-1, height * width, 3])

	
	
def findLocalPlanes(planes, planeMasks):
    batchSize = int(planeMasks.shape[0])
    height = int(planeMasks.shape[1])
    width = int(planeMasks.shape[2])
    numPlanes = int(planeMasks.shape[3])
    
    maskWidth = 16
    maskHeight = 16

    urange = tf.reshape(tf.range(width, dtype=tf.float32), [1, -1, 1])
    planeXs = tf.reduce_max(planeMasks, axis=1)
    planeMinX = float(width) - tf.reduce_max(planeXs * (float(width) - urange), axis=1)
    planeMaxX = tf.reduce_max(planeXs * urange, axis=1)

    vrange = tf.reshape(tf.range(height, dtype=tf.float32), [1, -1, 1])
    planeYs = tf.reduce_max(planeMasks, axis=2)
    planeMinY = float(height) - tf.reduce_max(planeYs * (float(height) - vrange), axis=1)
    planeMaxY = tf.reduce_max(planeYs * vrange, axis=1)


    localPlaneWidthThreshold = 64
    localPlaneHeightThreshold = 64
    localPlaneMask = tf.logical_and(tf.less(planeMaxX - planeMinX, localPlaneWidthThreshold), tf.less(planeMaxY - planeMinY, localPlaneHeightThreshold))

    
    stride = 8
    boxSize = 64
    xs = tf.tile(tf.expand_dims(tf.range(width / stride, dtype=tf.float32) * stride + stride / 2, 0), [height / stride, 1])
    ys = tf.tile(tf.expand_dims(tf.range(height / stride, dtype=tf.float32) * stride + stride / 2, 1), [1, width / stride])
    gridMinX = tf.reshape(xs - boxSize / 2, [1, -1, 1])
    gridMaxX = tf.reshape(xs + boxSize / 2, [1, -1, 1])
    gridMinY = tf.reshape(ys - boxSize / 2, [1, -1, 1])
    gridMaxY = tf.reshape(ys + boxSize / 2, [1, -1, 1])
    
    gridMinX = tf.tile(gridMinX, [batchSize, 1, 1])
    gridMaxX = tf.tile(gridMaxX, [batchSize, 1, 1])
    gridMinY = tf.tile(gridMinY, [batchSize, 1, 1])
    gridMaxY = tf.tile(gridMaxY, [batchSize, 1, 1])

    padding = boxSize / 2 + 1
    padding = boxSize / 2 + 1
    paddedPlaneMasks = tf.concat([tf.zeros([batchSize, height, padding, numPlanes]), planeMasks, tf.zeros([batchSize, height, padding, numPlanes])], axis=2)
    paddedPlaneMasks = tf.concat([tf.zeros([batchSize, padding, width + padding * 2, numPlanes]), paddedPlaneMasks, tf.zeros([batchSize, padding, width + padding * 2, numPlanes])], axis=1)

    gridPlaneMasks = []
    for gridY in xrange(height / stride):
        for gridX in xrange(width / stride):
            localPlaneMasks = tf.slice(paddedPlaneMasks, [0, gridY * stride + stride / 2 - boxSize / 2 + padding, gridX * stride + stride / 2 - boxSize / 2 + padding, 0], [batchSize, boxSize, boxSize, numPlanes])
            gridPlaneMasks.append(tf.image.resize_bilinear(localPlaneMasks, [maskHeight, maskWidth]))
            continue
        continue
    gridPlaneMasks = tf.stack(gridPlaneMasks, axis=1)
    gridPlaneMasks = tf.reshape(gridPlaneMasks, [batchSize, -1, maskHeight * maskWidth, numPlanes])

    planeMinX = tf.expand_dims(planeMinX, 1)
    planeMaxX = tf.expand_dims(planeMaxX, 1)
    planeMinY = tf.expand_dims(planeMinY, 1)
    planeMaxY = tf.expand_dims(planeMaxY, 1)    
    intersection = tf.maximum(tf.minimum(gridMaxX, planeMaxX) - tf.maximum(gridMinX, planeMinX) + 1, 0.) * tf.maximum(tf.minimum(gridMaxY, planeMaxY) - tf.maximum(gridMinY, planeMinY) + 1, 0.)
    union = (gridMaxX - gridMinX + 1) * (gridMaxY - gridMinY + 1) + (planeMaxX - planeMinX + 1) * (planeMaxY - planeMinY + 1) - intersection
    IOU = intersection / union
    #maxIOUInds = tf.argmax(IOU, axis=1)
    #maxIOU = tf.reduce_max(IOU, axis=1)
    IOU = IOU * tf.expand_dims(tf.cast(localPlaneMask, tf.float32), 1)
    IOU = IOU * tf.one_hot(tf.argmax(IOU, 1), depth=IOU.shape[1], axis=1)
    IOUThreshold = 1.0 / pow(boxSize / stride, 2)
    activeGridMask = tf.one_hot(tf.argmax(IOU, 2), depth=IOU.shape[2], axis=2) * tf.cast(tf.greater(IOU, IOUThreshold), tf.float32)
    
    #activeGridMask = tf.one_hot(tf.ones((batchSize, IOU.shape[1]), dtype=tf.int32), depth=IOU.shape[2], axis=2)
    
    gridScores = tf.reduce_sum(activeGridMask, axis=2, keep_dims=True)
    activeGridMask = tf.expand_dims(activeGridMask, -1)
    gridPlanes = tf.reduce_sum(activeGridMask * tf.expand_dims(planes, 1), axis=2)
    gridMasks = tf.reduce_sum(activeGridMask * tf.transpose(gridPlaneMasks, [0, 1, 3, 2]), axis=2)

    gridScores = tf.reshape(gridScores, [batchSize, height / stride, width / stride, -1])
    gridPlanes = tf.reshape(gridPlanes, [batchSize, height / stride, width / stride, -1])
    gridMasks = tf.reshape(gridMasks, [batchSize, height / stride, width / stride, -1])
    
    return gridScores, gridPlanes, gridMasks


def findBoundaries(planes, planeMasks):
    height = int(planeMasks.shape[0])
    width = int(planeMasks.shape[1])
    
    planesD = tf.norm(planes, axis=1, keep_dims=True)
    planesD = tf.clip_by_value(planesD, 1e-5, 10)
    planesNormal = planes / planesD

    ND = tf.expand_dims(planesNormal, 0) * tf.expand_dims(planesD, 1)
    ND_diff = tf.reshape(ND - tf.transpose(ND, [1, 0, 2]), [-1, 3])
    coefX, coefY, coefZ = tf.unstack(ND_diff, axis=1)

    pixels = []
    focalLength = 517.97
    urange = tf.range(width, dtype=tf.float32) / focalLength
    ones = tf.ones(urange.shape)
    vs = (coefX * urange + coefY * ones) / coefZ
    pixels.append(tf.stack([tf.floor(vs), urange], axis=1))
    pixels.append(tf.stack([tf.ceil(vs), urange], axis=1))
    
    vrange = tf.range(height, dtype=tf.float32) / focalLength
    ones = tf.ones(vrange.shape)
    us = -(coefY * ones - coefZ * vrange) / coefX
    pixels.append(tf.stack([vrange, tf.floor(us)], axis=1))
    pixels.append(tf.stack([vrange, tf.ceil(us)], axis=1))

    v, u = tf.unstack(pixels, axis=1)
    validMask = tf.logical_and(tf.less(u, width), tf.less(v, height))
    validMask = tf.logical_and(validMask, tf.greater_equal(u, 0))
    validMask = tf.logical_and(validMask, tf.greater_equal(v, 0))
    
    pixels *= tf.expand_dims(invalidMask, -1)
    
    boundary = tf.sparse_to_dense(pixels, output_shape=[height, width], sparse_values=1)
    return boundary


def fitPlaneMasksModule(planes, depth, normal, width = 640, height = 480, numPlanes = 20, normalDotThreshold = np.cos(np.deg2rad(5)), distanceThreshold = 0.05, closing=True, one_hot=True):
    focalLength = 517.97
    urange = (tf.range(width, dtype=tf.float32) / (width + 1) - 0.5) / focalLength * 641
    urange = tf.tile(tf.reshape(urange, [1, -1]), [height, 1])
    vrange = (tf.range(height, dtype=tf.float32) / (height + 1) - 0.5) / focalLength * 481
    vrange = tf.tile(tf.reshape(vrange, [-1, 1]), [1, width])
        
    X = depth * tf.expand_dims(urange, -1)
    Y = depth
    Z = -depth * tf.expand_dims(vrange, -1)
    XYZ = tf.concat([X, Y, Z], axis=3)
    XYZ = tf.reshape(XYZ, [-1, height * width, 3])
    plane_parameters = planes
    planesD = tf.norm(plane_parameters, axis=2, keep_dims=True)
    planesNormal = tf.div(tf.negative(plane_parameters), tf.clip_by_value(planesD, 1e-4, 10))

    distance = tf.reshape(tf.abs(tf.matmul(XYZ, planesNormal, transpose_b=True) + tf.reshape(planesD, [-1, 1, numPlanes])), [-1, height, width, numPlanes])
    angle = tf.reshape(np.abs(tf.matmul(tf.reshape(normal, [-1, height * width, 3]), planesNormal, transpose_b=True)), [-1, height, width, numPlanes])

    planeMasks = tf.cast(tf.logical_and(tf.greater(angle, normalDotThreshold), tf.less(distance, distanceThreshold)), tf.float32)

    if closing:
        #morphological closing
        planeMasks = tf.nn.max_pool(planeMasks, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
        pass
    plane_mask = tf.reduce_max(planeMasks, axis=3, keep_dims=True)
    if one_hot:
        if closing:
            plane_mask = 1 - tf.nn.max_pool(1 - plane_mask, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
            pass
        #one-hot encoding
        planeMasks = tf.one_hot(tf.argmax(planeMasks * (distanceThreshold - distance), axis=3), depth=numPlanes) * plane_mask
        pass
    
    return planeMasks, plane_mask
    

def depthToNormalModule(depth):
    batchSize = int(depth.shape[0])
    height = int(depth.shape[1])
    width = int(depth.shape[2])
    
    focalLength = 517.97
    urange = (tf.range(width, dtype=tf.float32) / width - 0.5) / focalLength * 640
    urange = tf.tile(tf.reshape(urange, [1, -1]), [height, 1])
    vrange = (tf.range(height, dtype=tf.float32) / height - 0.5) / focalLength * 480
    vrange = tf.tile(tf.reshape(vrange, [-1, 1]), [1, width])
        
    X = depth * tf.expand_dims(urange, -1)
    Y = depth
    Z = -depth * tf.expand_dims(vrange, -1)
    XYZ = tf.concat([X, Y, Z], axis=3)
    #XYZ = tf.reshape(XYZ, [-1, height * width, 3])

    
    kernel_array = np.zeros((3, 3, 1, 4))
    kernel_array[0, 1, 0, 0] = 1
    kernel_array[1, 0, 0, 1] = 1
    kernel_array[2, 1, 0, 2] = 1
    kernel_array[1, 2, 0, 3] = 1
    kernel_array[1, 1, 0, 0] = -1
    kernel_array[1, 1, 0, 1] = -1
    kernel_array[1, 1, 0, 2] = -1
    kernel_array[1, 1, 0, 3] = -1
    kernel = tf.constant(kernel_array.reshape(-1), shape=kernel_array.shape, dtype=tf.float32)
    XYZ_diff = tf.nn.depthwise_conv2d(tf.pad(XYZ, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT'), tf.tile(kernel, [1, 1, 3, 1]), strides=[1, 1, 1, 1], padding='VALID')
    XYZ_diff = tf.reshape(XYZ_diff, [-1, height, width, 3, 4])
    XYZ_diff_2 = tf.concat([tf.slice(XYZ_diff, [0, 0, 0, 0, 1], [batchSize, height, width, 3, 3]), tf.slice(XYZ_diff, [0, 0, 0, 0, 0], [batchSize, height, width, 3, 1])], axis=4)
    XYZ_diff_1 = tf.unstack(XYZ_diff, axis=3)
    XYZ_diff_2 = tf.unstack(XYZ_diff_2, axis=3)

    normal_X = XYZ_diff_1[1] * XYZ_diff_2[2] - XYZ_diff_1[2] * XYZ_diff_2[1]
    normal_Y = XYZ_diff_1[2] * XYZ_diff_2[0] - XYZ_diff_1[0] * XYZ_diff_2[2]
    normal_Z = XYZ_diff_1[0] * XYZ_diff_2[1] - XYZ_diff_1[1] * XYZ_diff_2[0]

    normal_X = tf.reduce_sum(normal_X, axis=[3])
    normal_Y = tf.reduce_sum(normal_Y, axis=[3])
    normal_Z = tf.reduce_sum(normal_Z, axis=[3])
    normal = tf.stack([normal_X, normal_Y, normal_Z], axis=3)

    kernel_size = 5
    padding = (kernel_size - 1) / 2
    neighbor_kernel_array = gaussian(kernel_size)
    #neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
    neighbor_kernel_array /= neighbor_kernel_array.sum()
    neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
    neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])
    normal = tf.nn.depthwise_conv2d(tf.pad(normal, [[0, 0], [padding, padding], [padding, padding], [0, 0]], mode='REFLECT'), tf.tile(neighbor_kernel, [1, 1, 3, 1]), strides=[1, 1, 1, 1], padding='VALID')
    
    normal = normal / tf.norm(normal, axis=3, keep_dims=True)
    return normal

def findBoundaryModule(depth, normal, segmentation, plane_mask, max_depth_diff = 0.1, max_normal_diff = np.sqrt(2 * (1 - np.cos(np.deg2rad(20))))):
    kernel_size = 3
    padding = (kernel_size - 1) / 2
    neighbor_kernel_array = gaussian(kernel_size)
    neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
    neighbor_kernel_array /= neighbor_kernel_array.sum()
    neighbor_kernel_array *= -1
    neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 1
    neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
    neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])
        
    depth_diff = tf.abs(tf.nn.depthwise_conv2d(depth, neighbor_kernel, strides=[1, 1, 1, 1], padding='VALID'))
    depth_diff = tf.pad(depth_diff, paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]])
    max_depth_diff = 0.1
    depth_boundary = tf.greater(depth_diff, max_depth_diff)

    normal_diff = tf.norm(tf.nn.depthwise_conv2d(normal, tf.tile(neighbor_kernel, [1, 1, 3, 1]), strides=[1, 1, 1, 1], padding='VALID'), axis=3, keep_dims=True)
    normal_diff = tf.pad(normal_diff, paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]])
    max_normal_diff = np.sqrt(2 * (1 - np.cos(np.deg2rad(20))))
    normal_boundary = tf.greater(normal_diff, max_normal_diff)

    plane_region = tf.nn.max_pool(plane_mask, ksize=[1, kernel_size, kernel_size, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
    segmentation_eroded = 1 - tf.nn.max_pool(1 - segmentation, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
    plane_region -= tf.reduce_max(segmentation_eroded, axis=3, keep_dims=True)
    boundary = tf.cast(tf.logical_or(depth_boundary, normal_boundary), tf.float32) * plane_region
    #boundary = plane_region
    #smooth_boundary = tf.cast(tf.less_equal(depth_diff, max_depth_diff), tf.float32) * boundary
    smooth_boundary = tf.cast(tf.logical_and(normal_boundary, tf.less_equal(depth_diff, max_depth_diff)), tf.float32)
    smooth_boundary = tf.nn.max_pool(smooth_boundary, ksize=[1, kernel_size, kernel_size, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool') * boundary
    #smooth_boundary = smooth_boundary * boundary
    boundary_gt = tf.concat([smooth_boundary, boundary - smooth_boundary], axis=3)
    return boundary_gt


def findBoundaryModuleSmooth(depth, segmentation, plane_mask, smooth_boundary, max_depth_diff = 0.1, max_normal_diff = np.sqrt(2 * (1 - np.cos(np.deg2rad(20))))):
    kernel_size = 3
    padding = (kernel_size - 1) / 2
    neighbor_kernel_array = gaussian(kernel_size)
    neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
    neighbor_kernel_array /= neighbor_kernel_array.sum()
    neighbor_kernel_array *= -1
    neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 1
    neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
    neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])
        
    depth_diff = tf.abs(tf.nn.depthwise_conv2d(depth, neighbor_kernel, strides=[1, 1, 1, 1], padding='VALID'))
    depth_diff = tf.pad(depth_diff, paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]])
    max_depth_diff = 0.1
    depth_boundary = tf.greater(depth_diff, max_depth_diff)


    plane_region = tf.nn.max_pool(plane_mask, ksize=[1, kernel_size, kernel_size, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
    segmentation_eroded = 1 - tf.nn.max_pool(1 - segmentation, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
    plane_region -= tf.reduce_max(segmentation_eroded, axis=3, keep_dims=True)
    occlusion_boundary = tf.cast(depth_boundary, tf.float32) * plane_region
    #boundary = plane_region
    #smooth_boundary = tf.cast(tf.less_equal(depth_diff, max_depth_diff), tf.float32) * boundary
    smooth_boundary = smooth_boundary * plane_region
    smooth_boundary_dilated = tf.nn.max_pool(smooth_boundary, ksize=[1, kernel_size, kernel_size, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool') * plane_region
    #smooth_boundary = smooth_boundary * boundary
    boundary_gt = tf.concat([smooth_boundary, tf.maximum(occlusion_boundary - smooth_boundary_dilated, 0)], axis=3)
    return boundary_gt


def crfModule(segmentations, planes, non_plane_depth, info, numOutputPlanes=20, numIterations=20, kernel_size = 9):
    width = int(segmentations.shape[2])
    height = int(segmentations.shape[1])
    
    #maxDepthDiff = tf.Variable(0.3)
    #sigmaDepthDiff = tf.Variable(0.5)
    maxDepthDiff = 0.3
    sigmaDepthDiff = 0.5

    plane_parameters = tf.reshape(planes, (-1, 3))
    plane_depths = planeDepthsModule(plane_parameters, width, height, info)
    plane_depths = tf.transpose(tf.reshape(plane_depths, [height, width, -1, numOutputPlanes]), [2, 0, 1, 3])
    all_depths = tf.concat([plane_depths, non_plane_depth], axis=3)
    
    refined_segmentation = segmentations
    for _ in xrange(numIterations):
        refined_segmentation = meanfieldModule(refined_segmentation, all_depths, numOutputPlanes=numOutputPlanes + 1, sigmaDepthDiff=sigmaDepthDiff, kernel_size = kernel_size)
        continue
    return refined_segmentation

def divideLayers(segmentations, planes, non_plane_mask, info, num_planes, numOutputPlanes_0=5, validAreaRatio=0.95, distanceThreshold=0.05):
    batchSize = int(planes.shape[0])    
    numOutputPlanes = int(planes.shape[1])
    width = int(segmentations.shape[2])
    height = int(segmentations.shape[1])
    
    plane_parameters = tf.reshape(planes, (-1, 3))
    plane_depths = planeDepthsModule(plane_parameters, width, height, info)
    plane_depths = tf.transpose(tf.reshape(plane_depths, [height, width, -1, numOutputPlanes]), [2, 0, 1, 3])
    depth = tf.reduce_sum(plane_depths * segmentations[:, :, :, :numOutputPlanes], axis=3, keep_dims=True)
    #non_plane_mask = segmentations[:, :, :, numOutputPlanes:numOutputPlanes+1]
    
    background_mask = tf.logical_or(tf.logical_or(tf.less(plane_depths, 1e-4), tf.greater(plane_depths, depth - distanceThreshold)), tf.cast(non_plane_mask, tf.bool))
    background_planes = tf.greater(tf.reduce_mean(tf.cast(background_mask, tf.float32), axis=[1, 2]), validAreaRatio)
    validPlaneMask = tf.less(tf.tile(tf.expand_dims(tf.range(numOutputPlanes), 0), [batchSize, 1]), tf.expand_dims(num_planes, -1))
    background_planes = tf.logical_and(background_planes, validPlaneMask)
    background_planes = tf.cast(background_planes, tf.float32)
    plane_areas = tf.reduce_sum(segmentations[:, :, :, :numOutputPlanes], axis=[1, 2])
    
    layer_plane_areas_0 = plane_areas * background_planes    
    areas, sortInds = tf.nn.top_k(layer_plane_areas_0, k=numOutputPlanes_0)
    sortMap = tf.one_hot(sortInds, depth=numOutputPlanes, axis=1)
    validMask = tf.cast(tf.greater(areas, 1e-4), tf.float32)
    sortMap *= tf.expand_dims(validMask, 1)
    layer_segmentations_0 = tf.reshape(tf.matmul(tf.reshape(segmentations, [batchSize, height * width, -1]), sortMap), [batchSize, height, width, -1])
    layer_planes_0 = tf.transpose(tf.matmul(planes, sortMap, transpose_a=True), [0, 2, 1])


    layer_plane_areas_1 = plane_areas * (1 - background_planes)
    areas, sortInds = tf.nn.top_k(layer_plane_areas_1, k=numOutputPlanes - numOutputPlanes_0)
    sortMap = tf.one_hot(sortInds, depth=numOutputPlanes, axis=1)
    validMask = tf.cast(tf.greater(areas, 1e-4), tf.float32)
    sortMap *= tf.expand_dims(validMask, 1)
    layer_segmentations_1 = tf.reshape(tf.matmul(tf.reshape(segmentations, [batchSize, height * width, -1]), sortMap), [batchSize, height, width, -1])
    layer_planes_1 = tf.transpose(tf.matmul(planes, sortMap, transpose_a=True), [0, 2, 1])
    
    
    return tf.concat([layer_segmentations_0, layer_segmentations_1], axis=3), tf.concat([layer_planes_0, layer_planes_1], axis=1)



def calcMessages(planeSegmentations, planeDepths, planesY, numOutputPlanes = 21, coef = [1, 1, 1], beta = 1, iteration = 0, maxDepthDiff = 0.2, varDepthDiff = 0.5, kernel_size = 9):
    #images, varImageDiff
    batchSize = int(planeSegmentations.shape[0])
    height = int(planeSegmentations.shape[1])
    width = int(planeSegmentations.shape[2])


    n2 = tf.pow(tf.reshape(planesY, [batchSize, 1, 1, -1]), 2)
    d2n2s = tf.reduce_sum(tf.pow(planeDepths, 2) * n2 * planeSegmentations, axis=-1, keep_dims=True)
    dnsd = tf.reduce_sum(planeDepths * n2 * planeSegmentations, axis=-1, keep_dims=True) * planeDepths
    n2sd2 = tf.reduce_sum(n2 * planeSegmentations, axis=-1, keep_dims=True) * tf.pow(planeDepths, 2)

    messages = d2n2s - 2 * dnsd + n2sd2

    maxDepthDiff = 0.2
    messages = tf.minimum(messages / pow(maxDepthDiff, 2), 1)
    
    # vertical_padding = tf.zeros((batchSize, height, 1, numOutputPlanes))
    # horizontal_padding = tf.zeros((batchSize, height, 1, numOutputPlanes))    


    # neighbor_kernel_array = gaussian(kernel_size)
    # neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
    # neighbor_kernel_array /= neighbor_kernel_array.sum()
    # neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
    # neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])
    
    # messages = tf.nn.depthwise_conv2d(messages, tf.tile(neighbor_kernel, [1, 1, numOutputPlanes, 1]), strides=[1, 1, 1, 1], padding='SAME')

    return messages


def crfrnnModule(inputs, image_dims, num_classes, theta_alpha, theta_beta, theta_gamma, num_iterations):
    custom_module = tf.load_op_library('./cpp/high_dim_filter.so')
    import high_dim_filter_grad  # Register gradients for the custom op

    weights = np.load('weights.npy')
    weights = [weights[0], weights[1], weights[2]]
    spatial_ker_weights = tf.Variable(weights[0][:num_classes, :num_classes], name='spatial_ker_weights', trainable=True)
    bilateral_ker_weights = tf.Variable(weights[1][:num_classes, :num_classes], name='bilateral_ker_weights', trainable=True)
    compatibility_matrix = tf.Variable(weights[2][:num_classes, :num_classes], name='compatibility_matrix', trainable=True)
    

    batchSize = int(inputs[0].shape[0])
    c, h, w = num_classes, image_dims[0], image_dims[1]
    all_ones = np.ones((c, h, w), dtype=np.float32)

    outputs = []
    for batchIndex in xrange(batchSize):
        unaries = tf.transpose(inputs[0][batchIndex, :, :, :], perm=(2, 0, 1))
        rgb = tf.transpose(inputs[1][batchIndex, :, :, :], perm=(2, 0, 1))


        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                          theta_gamma=theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                            theta_alpha=theta_alpha,
                                                            theta_beta=theta_beta)
        q_values = unaries

        for i in range(num_iterations):
            softmax_out = tf.nn.softmax(q_values, dim=0)

            # Spatial filtering
            spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                        theta_gamma=theta_gamma)
            spatial_out = spatial_out / spatial_norm_vals

            # Bilateral filtering
            bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                          theta_alpha=theta_alpha,
                                                          theta_beta=theta_beta)
            bilateral_out = bilateral_out / bilateral_norm_vals

            # Weighting filter outputs
            message_passing = (tf.matmul(spatial_ker_weights,
                                         tf.reshape(spatial_out, (c, -1))) +
                               tf.matmul(bilateral_ker_weights,
                                         tf.reshape(bilateral_out, (c, -1))))

            # Compatibility transform
            pairwise = tf.matmul(compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(pairwise, (c, h, w))
            q_values = unaries - pairwise
            continue
        outputs.append(tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1)))
        continue
    outputs = tf.concat(outputs, axis=0)
    return outputs
