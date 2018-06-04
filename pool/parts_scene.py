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
  
class PartsScene():
  def __init__(self, index):
    #self.depth = cv2.imread('dump/' + str(index) + '_depth_pred.png').astype(np.float32) / 255 * 10
    self.depth = np.load('dump/' + str(index) + '_depth.npy')
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
    

  def generateEggModel(self):
  
    self.planeNPs = []
    self.planeCenters = []

    print(self.numPlanes)
    for planeIndex in xrange(self.numPlanes):
      mask = (self.segmentation == planeIndex).astype(np.uint8) * 255
      #cv2.imwrite('test/mask_' + str(planeIndex) + '.png', mask)
      contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      plane = self.planes[planeIndex]
      planeD = np.linalg.norm(plane)
      planeNormal = plane / planeD

      for contour in contours:
        data = EggData()
        model = EggGroup('model')
        data.addChild(model)

        vp = EggVertexPool('plane_vertex')
        model.addChild(vp)
        
        planeGroup = EggGroup('plane')
        model.addChild(planeGroup)
        poly = EggPolygon()
        planeGroup.addChild(poly)

        poly.setTexture(self.imageTexture.getEggTexture())
        poly.setMaterial(self.imageTexture.getEggMaterial())

        contour = contour.astype(np.float32)
        #u = (contour[:, 0, 0] - self.width / 2) / self.width * 640 / self.focalLength
        #v = -(contour[:, 0, 1] - self.height / 2) / self.height * 480 / self.focalLength
        u = (contour[:, 0, 0].astype(np.float32) / self.width * self.info[16] - self.camera['cx']) / self.camera['fx']
        v = -(contour[:, 0, 1].astype(np.float32) / self.height * self.info[17] - self.camera['cy']) / self.camera['fy']

        ranges = np.stack([u, np.ones(u.shape), v], axis=1)
        depth = planeD / np.dot(ranges, planeNormal)
        XYZ = ranges * np.expand_dims(depth, -1)
        center = XYZ.mean(0)
        #print(contour)
        #print(XYZ)
        #exit(1)
        for vertexIndex, uv in enumerate(contour):
          vertex = EggVertex()
          X, Y, Z = XYZ[vertexIndex]
          vertex.setPos(Point3D(X - center[0], Y - center[1], Z - center[2]))
          u, v = uv[0]
          vertex.setUv(Point2D(u / self.width, 1 - v / self.height))
          poly.addVertex(vp.addVertex(vertex))
          continue
        scene = NodePath(loadEggData(data))
        self.planeNPs.append(scene)
        self.planeCenters.append(center)
        continue
    return self.planeNPs, self.planeCenters


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
