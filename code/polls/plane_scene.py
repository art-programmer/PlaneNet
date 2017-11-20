from panda3d.egg import *
from panda3d.core import *
from obj2egg import ObjMaterial
from copy import deepcopy
import numpy as np
import cv2
import copy
from direct.gui.OnscreenImage import OnscreenImage

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import getCameraFromInfo

def calcDistance(point_1, point_2):
  return pow(pow(point_1[0] - point_2[0], 2) + pow(point_1[1] - point_2[1], 2), 0.5)

def calcLineDim(line, lineWidth = -1):
  if abs(line[0][0] - line[1][0]) > abs(line[0][1] - line[1][1]):
    if lineWidth < 0 or abs(line[0][1] - line[1][1]) <= lineWidth:
      return 0
    pass
  elif abs(line[0][0] - line[1][0]) < abs(line[0][1] - line[1][1]):
    if lineWidth < 0 or abs(line[0][0] - line[1][0]) <= lineWidth:
      return 1
  else:
    return -1
  
class PlaneScene():
  def __init__(self, index):
    #self.depth = cv2.imread('dump/' + str(index) + '_depth_pred.png').astype(np.float32) / 255 * 10
    self.depth = np.load('dump/' + str(index) + '_depth.npy')
    #cv2.imwrite('dump/alpha_0.5.png', np.zeros(self.depth[:, :, 0].shape).astype(np.uint8))
    self.segmentation = np.load('dump/' + str(index) + '_segmentation.npy')

    width = 640
    height = 480
    self.depth = cv2.resize(self.depth, (width, height))
    self.segmentation = cv2.resize(self.segmentation, (width, height), interpolation=cv2.INTER_NEAREST)
    self.planes = np.load('dump/' + str(index) + '_planes.npy')
    self.numPlanes = self.planes.shape[0]

    self.imageTexture = ObjMaterial()
    self.imageTexture.name = 'image'
    self.imageTexture.put('map_Kd', 'dump/' + str(index) + '_image.png')
    self.width = self.depth.shape[1]
    self.height = self.depth.shape[0]
    self.info = np.load('dump/' + str(index) + '_info.npy')
    self.camera = getCameraFromInfo(self.info)
    self.scene_index = index
    self.calcHorizontalPlanes()
    return

  def addRectangle(self, parent):
    planesGroup = EggGroup('planes')
    parent.addChild(planesGroup)
    vp = EggVertexPool('plane_vertex')
    parent.addChild(vp)
    
    p0 = Point3D(-10, 1, 0)
    p1 = Point3D(-10, 10, 0)
    p2 = Point3D(10, 1, 0)
    p3 = Point3D(10, 10, 0)    
    # p0 = Point3D(-10, , 0)
    # p1 = Point3D(-10, 100, 0)
    # p3 = Point3D(10, 100, 0)
    # p2 = Point3D(10, 90, 0)
    
    planeGroup = EggGroup('plane')
    planesGroup.addChild(planeGroup)
    poly = EggPolygon()
    planeGroup.addChild(poly)
    vertex = EggVertex()
    vertex.setPos(p0)
    vertex.setUv(Point2D(0, 0))
    poly.addVertex(vp.addVertex(vertex))
    vertex = EggVertex()
    vertex.setPos(p1)
    vertex.setUv(Point2D(0, 1))
    poly.addVertex(vp.addVertex(vertex))
    vertex = EggVertex()
    vertex.setPos(p2)
    vertex.setUv(Point2D(1, 1))
    poly.addVertex(vp.addVertex(vertex))

    
    poly = EggPolygon()
    planeGroup.addChild(poly)
    
    vertex = EggVertex()
    vertex.setPos(p1)
    vertex.setUv(Point2D(0, 1))
    poly.addVertex(vp.addVertex(vertex))
    vertex = EggVertex()
    vertex.setPos(p2)
    vertex.setUv(Point2D(1, 1))
    poly.addVertex(vp.addVertex(vertex))
    vertex = EggVertex()
    vertex.setPos(p3)
    vertex.setUv(Point2D(1, 0))
    poly.addVertex(vp.addVertex(vertex))

    # vertex = EggVertex()
    # vertex.setPos(p2)
    # vertex.setUv(Point2D(1, 1))
    # poly.addVertex(vp.addVertex(vertex))
    
    return
    

  def generatePlanes(self, parent):
    planesGroup = EggGroup('planes')
    parent.addChild(planesGroup)
    vp = EggVertexPool('plane_vertex')
    parent.addChild(vp)

    for planeIndex in xrange(self.numPlanes):
      mask = (self.segmentation == planeIndex).astype(np.uint8) * 255
      cv2.imwrite('test/mask_' + str(planeIndex) + '.png', mask)
      contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      plane = self.planes[planeIndex]
      planeD = np.linalg.norm(plane)
      planeNormal = plane / planeD
      for contour in contours:
        planeGroup = EggGroup('plane')
        planesGroup.addChild(planeGroup)
        poly = EggPolygon()
        planeGroup.addChild(poly)

        poly.setTexture(self.imageTexture.getEggTexture())
        poly.setMaterial(self.imageTexture.getEggMaterial())

        contour = contour.astype(np.float32)
        u = (contour[:, 0, 0].astype(np.float32) / self.width * self.info[16] - self.camera['cx']) / self.camera['fx']
        v = -(contour[:, 0, 1].astype(np.float32) / self.height * self.info[17] - self.camera['cy']) / self.camera['fy']
        ranges = np.stack([u, np.ones(u.shape), v], axis=1)
        depth = planeD / np.dot(ranges, planeNormal)
        XYZ = ranges * np.expand_dims(depth, -1)
        #print(contour)
        #print(XYZ)
        #exit(1)
        for vertexIndex, uv in enumerate(contour):
          vertex = EggVertex()
          X, Y, Z = XYZ[vertexIndex]
          vertex.setPos(Point3D(X, Y, Z))
          u, v = uv[0]
          vertex.setUv(Point2D(u / self.width, 1 - v / self.height))
          poly.addVertex(vp.addVertex(vertex))
          continue
        continue
      continue
    return


  def generateRectangle(self, parent):    
    planesGroup = EggGroup('planes')
    parent.addChild(planesGroup)
    vp = EggVertexPool('plane_vertex')
    parent.addChild(vp)

    poly = EggPolygon()
    planesGroup.addChild(poly)


    w = 0.5
    p0 = Point3D(-w / 2, 0, -w / 2)
    p1 = Point3D(-w / 2, 0, w / 2)
    p2 = Point3D(w / 2, 0, w / 2)
    p3 = Point3D(w / 2, 0, -w / 2)    
  
    
    poly.setTexture(self.plateTexture.getEggTexture())
    poly.setMaterial(self.plateTexture.getEggMaterial())
    vertex = EggVertex()
    vertex.setPos(Point3D(0, 1, 0))
    vertex.setUv(Point2D(0, 0))
    poly.addVertex(vp.addVertex(vertex))
    vertex = EggVertex()
    vertex.setPos(Point3D(0, 1, 1))
    vertex.setUv(Point2D(0, 1))
    poly.addVertex(vp.addVertex(vertex))
    vertex = EggVertex()
    vertex.setPos(Point3D(1, 1, 1))
    vertex.setUv(Point2D(1, 1))
    poly.addVertex(vp.addVertex(vertex))
    vertex = EggVertex()
    vertex.setPos(Point3D(1, 1, 0))
    vertex.setUv(Point2D(1, 0))
    poly.addVertex(vp.addVertex(vertex))
    
    return

  def addCollisionPolygons(self, scene):
    
    polygons = scene.findAllMatches("**/plane")
    mesh = BulletTriangleMesh()      
    for polygon in polygons:
      #cNode = scene.attachNewNode(CollisionNode('plane_solid'))
      #cNode.node().addSolid(CollisionPolygon(polygon))
      #polygon.setCollideMask(BitMask32.bit(1))
      node = polygon.node()
      print(node.getNumGeoms())
      for i in xrange(node.getNumGeoms()):
        geom = node.getGeom(i)
        mesh.addGeom(geom)
        continue
      continue

  def test(self, scene):
    groundMask=BitMask32(0b1)
    parent = NodePath('cGeomConversionParent') 
    for c in incomingNode.findAllMatches('**/+GeomNode'): 
        if relativeTo:
            xform=c.getMat(relativeTo).xformPoint
        else:
            xform=c.getMat().xformPoint
        gni = 0 
        geomNode = c.node() 
        for g in range(geomNode.getNumGeoms()): 
            geom = geomNode.getGeom(g).decompose() 
            vdata = geom.getVertexData() 
            vreader = GeomVertexReader(vdata, 'vertex') 
            cChild = CollisionNode('cGeom-%s-gni%i' % (c.getName(), gni)) 
            
            gni += 1 
            for p in range(geom.getNumPrimitives()): 
                prim = geom.getPrimitive(p) 
                for p2 in range(prim.getNumPrimitives()): 
                    s = prim.getPrimitiveStart(p2) 
                    e = prim.getPrimitiveEnd(p2) 
                    v = [] 
                    for vi in range (s, e): 
                        vreader.setRow(prim.getVertex(vi)) 
                        v.append (xform(vreader.getData3f())) 
                    colPoly = CollisionPolygon(*v) 
                    cChild.addSolid(colPoly) 

            n=parent.attachNewNode (cChild) 
            #n.show()
            
    return parent

  
  def generateEggModel(self):
    data = EggData()
    model = EggGroup('model')
    data.addChild(model)
    self.generatePlanes(model)
    #self.generateRectangle(model)
    data.writeEgg(Filename("test/plane.egg"))
    scene = NodePath(loadEggData(data))
    #self.addCollisionPolygons(scene)
    
    return scene  

    

  def getPlaneTriangles(self):
    from skimage import measure
    
    planeTriangles = []
    planeNormals = []    
    horizontalPlaneTriangles = []
    
    for planeIndex in xrange(self.numPlanes):
      mask = (self.segmentation == planeIndex).astype(np.uint8) * 255
      mask_ori = mask.copy()
      #contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      #contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
      masks = measure.label(mask.astype(np.int32), background=0)
      contours = []
      for maskIndex in xrange(masks.min() + 1, masks.max() + 1):
        mask = masks == maskIndex
        contour_mask = mask - np.logical_and(np.logical_and(np.roll(mask, shift=1, axis=0), np.roll(mask, shift=-1, axis=0)), np.logical_and(np.roll(mask, shift=1, axis=1), np.roll(mask, shift=-1, axis=1)))
        contour_v, contour_u = contour_mask.nonzero()
        contours.append(np.stack([contour_u, contour_v], axis=1))
        continue
        
        
      plane = self.planes[planeIndex]
      planeD = np.linalg.norm(plane)
      planeNormal = plane / np.maximum(planeD, 1e-4)

      # cv2.imwrite('test/mask.png', mask_ori)
      # #print(len(contours))

      # mask_ori = np.stack([mask_ori, mask_ori, mask_ori], 2)
      # count = 0
      # for contour in contours:
      #   count += contour.shape[0]
      #   for uv in contour:
      #     #uv = uv[0]
      #     mask_ori[uv[1]][uv[0]] = np.array([255, 0, 0])
      #     continue
      #   continue
      # cv2.imwrite('test/mask_contour.png', mask_ori)
      # if planeIndex == 1:
      #   exit(1)

      indices = np.arange(self.width * self.height).astype(np.float32)
      us = indices % self.width
      us = us / self.width * self.info[16] - self.camera['cx']
      vs = indices / self.width      
      vs = -(vs / self.height * self.info[17] - self.camera['cy'])
      ranges = np.stack([us / self.camera['fx'], np.ones(us.shape), vs / self.camera['fy']], axis=1)
      #print(ranges)
      #print(np.dot(ranges, planeNormal).shape)
      #print(np.dot(ranges, planeNormal))
      #print(ranges)
      #exit(1)
      depth = planeD / np.tensordot(ranges, planeNormal, axes=([1], [0]))
      XYZ = ranges * np.expand_dims(depth, -1)
      XYZ = XYZ.reshape((self.height, self.width, 3))
      for contour in contours:
        contour = contour.astype(np.float32)[::20]
        if contour.shape[0] < 3:
          continue
        rect = (0, 0, self.width, self.height)
        subdiv = cv2.Subdiv2D(rect)

        for point in contour:
          subdiv.insert((point[0], point[1]))
          continue
        triangleList = subdiv.getTriangleList()

        #print(contour)                
        #print(triangleList)
        #exit(1)
        for triangle2D in triangleList:
          triangle = []
          for vertexIndex in xrange(3):
            x = int(triangle2D[vertexIndex * 2 + 0])
            y = int(triangle2D[vertexIndex * 2 + 1])
            #print(x, y)
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
              continue
            triangle.append(XYZ[y][x])
            continue
          if len(triangle) == 3:
            #print(triangle)
            if np.dot(np.cross(planeNormal, triangle[1] - triangle[0]), triangle[2] - triangle[0]) > 0:
              triangle = [triangle[0], triangle[2], triangle[1]]
              pass
            if planeIndex in self.horizontalPlanes:
              horizontalPlaneTriangles.append(triangle)
            else:
              planeTriangles.append(triangle)
              pass
            #planeNormals.append(planeNormal)
            pass
          continue
      continue
    planeTriangles = np.array(planeTriangles)
    #planeNormals = np.array(planeNormals)
    np.save('dump/' + str(self.scene_index) + '_plane_triangles.npy', planeTriangles)
    #np.save('dump/' + str(self.scene_index) + '_plane_normals.npy', planeNormals)
    return planeTriangles, horizontalPlaneTriangles, self.gravityDirection
  

  def getPlaneGeometries(self):
    if os.path.exists('dump/' + str(self.scene_index) + '_plane_triangles.npy'):
      print('loading')
      planeTriangles = np.load('dump/' + str(self.scene_index) + '_plane_triangles.npy')
      planeNormals =  np.load('dump/' + str(self.scene_index) + '_plane_normals.npy')
      return planeTriangles, planeNormals
      pass
    
    planeNormals = []
    planeTriangles = []
    for planeIndex in xrange(self.numPlanes):
      mask = (self.segmentation == planeIndex).astype(np.uint8) * 255
      #mask_ori = mask.copy()
      #contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      plane = self.planes[planeIndex]
      planeD = np.linalg.norm(plane)
      planeNormal = plane / np.maximum(planeD, 1e-4)

      #cv2.imwrite('test/mask.png', mask)
      #v, u = mask.nonzero()
      u = np.arange(self.width * self.height) % self.width
      v = np.arange(self.width * self.height) / self.width
      u = u.astype(np.float32) / self.width * self.info[16] - self.camera['cx']
      v = -(v.astype(np.float32) / self.height * self.info[17] - self.camera['cy'])
      ranges = np.stack([u / self.camera['fx'], np.ones(u.shape), v / self.camera['fy']], axis=1)
      depth = planeD / np.dot(ranges, planeNormal)
      XYZ = ranges * np.expand_dims(depth, -1)
      XYZ = XYZ.reshape((self.height, self.width, 3))

      triangles = []
      for pixel in mask.reshape(-1).nonzero()[0]:
        x = pixel % self.width
        y = pixel / self.width
        for neighbors in [((x - 1, y), (x, y - 1)), ((x - 1, y), (x, y + 1)), ((x + 1, y), (x, y - 1)), ((x + 1, y), (x, y + 1))]:
          valid = True
          for neighbor in neighbors:
            if neighbor[0] < 0 or neighbor[0] >= self.width or neighbor[1] < 0 or neighbor[1] >= self.height or mask[neighbor[1]][neighbor[0]] == False:
              valid = False
              break
            continue
          if valid:
            triangle = [XYZ[y][x]]
            for neighbor in neighbors:
              triangle.append(XYZ[neighbor[1], neighbor[0]])
              continue
            triangles.append(triangle)
            pass
          continue
        continue
      planeTriangles.append(triangles)
      planeNormals.append(planeNormal)
      continue

    planeTriangles = np.array(planeTriangles)
    #planeNormals = np.array(planeNormals)
    #np.save('dump/' + str(self.scene_index) + '_plane_triangles.npy', planeTriangles)
    #np.save('dump/' + str(self.scene_index) + '_plane_normals.npy', planeNormals)
    return planeTriangles, planeNormals
  

  def calcHorizontalPlanes(self):
    from sklearn.cluster import KMeans
    
    planesD = np.linalg.norm(self.planes, axis=-1, keepdims=True)
    normals = self.planes / np.maximum(planesD, 1e-4)
    
    normals[normals[:, 1] < 0] *= -1    

    kmeans = KMeans(n_clusters=3).fit(normals)
    dominantNormals = kmeans.cluster_centers_
    dominantNormals = dominantNormals / np.maximum(np.linalg.norm(dominantNormals, axis=-1, keepdims=True), 1e-4)

    planeClusters = kmeans.predict(normals)
    
    horizontalNormalIndex = np.argmax(np.abs(dominantNormals[:, 2]))
    self.gravityDirection = dominantNormals[horizontalNormalIndex]
    self.horizontalPlanes = (planeClusters == horizontalNormalIndex).nonzero()[0]
    if self.gravityDirection[2] > 0:
      self.gravityDirection *= -1
      pass

    print(self.horizontalPlanes)
    print(self.gravityDirection)
    return
    
  def getHorizontalPlanes(self):
    return self.gravityDirection, self.horizontalPlanes
