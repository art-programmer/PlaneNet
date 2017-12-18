#!/usr/bin/env python

# Author: Shao Zhang, Phil Saltzman
# Last Updated: 2015-03-13
#
# This tutorial shows how to detect and respond to collisions. It uses solids
# create in code and the egg files, how to set up collision masks, a traverser,
# and a handler, how to detect collisions, and how to dispatch function based
# on the collisions. All of this is put together to simulate a labyrinth-style
# game

from direct.showbase.ShowBase import ShowBase
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionHandlerQueue, CollisionRay, CollisionPolygon, CollisionSphere, CollisionTube
from panda3d.core import Material, LRotationf, NodePath
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import TextNode
from panda3d.core import LVector3, BitMask32
from panda3d.core import PerspectiveLens, Vec3, Point3
from panda3d.core import CardMaker
from direct.gui.OnscreenText import OnscreenText
from direct.interval.MetaInterval import Sequence, Parallel
from direct.interval.LerpInterval import LerpFunc
from direct.interval.FunctionInterval import Func, Wait
from direct.task.Task import Task
from plane_scene import PlaneScene
from parts_scene import PartsScene
#from panda3d.bullet import BulletWorld, BulletPlaneShape, BulletRigidBodyNode, BulletBoxShape, BulletTriangleMesh, BulletTriangleMeshShape, BulletSphereShape, BulletGhostNode
import sys
import numpy as np

# Some constants for the program
ACCEL = 70         # Acceleration in ft/sec/sec
MAX_SPEED = 5      # Max speed in ft/sec
MAX_SPEED_SQ = MAX_SPEED ** 2  # Squared to make it easier to use lengthSquared
# Instead of length


class BallInMazeDemo(ShowBase):

    def __init__(self):
        # Initialize the ShowBase class from which we inherit, which will
        # create a window and set up everything we need for rendering into it.
        ShowBase.__init__(self)
        base.setBackgroundColor(0, 0, 0)
        
        self.accept("escape", sys.exit)  # Escape quits
        self.disableMouse()
        camera.setPosHpr(0, 0, 0, 0, 0, 0)

        lens = PerspectiveLens()
        lens.setFov(90, 60)
        lens.setNear(0.01)
        lens.setFar(100000)
        self.cam.node().setLens(lens)

        self.ballSize = 0.025
        self.cueLength = 0.2
        # self.ballRoot = render.attachNewNode("ballRoot")
        # #self.ball = loader.loadModel("models/ball")
        # self.ball = loader.loadModel("models/ball_0_center.egg")
        # #self.ball = loader.loadModel("models/ball.dae")
        # self.ball.setScale(ballSize, ballSize, ballSize)
        # self.ball.reparentTo(self.ballRoot)
        # #print(self.ball.getBounds())
        # #exit(1)
        # #self.ballSphere = self.ball.find("**/ball")
        # #print(self.ball.getScale()[0])
        # cs = CollisionSphere(0, 0, 0, 1)
        # self.ballSphere = self.ball.attachNewNode(CollisionNode('ball'))
        # self.ballSphere.node().addSolid(cs)
        
        # self.ballSphere.node().setFromCollideMask(BitMask32.bit(0))
        # self.ballSphere.node().setIntoCollideMask(BitMask32.bit(1))


        self.sceneIndex = 2
        self.planeInfo = PlaneScene(self.sceneIndex)
        
        self.planeScene = self.planeInfo.generateEggModel()
        self.planeScene.setTwoSided(True)        
        self.planeScene.reparentTo(render)
        self.planeScene.hide()
        
        planeTriangles, horizontalPlaneTriangles, self.gravityDirection = self.planeInfo.getPlaneTriangles()

        
        self.ballRoots = []
        self.balls = []
        self.ballSpheres = []
        self.ballGroundRays = []
        for ballIndex in xrange(3):
            ballRoot = render.attachNewNode("ballRoot_" + str(ballIndex))
            ball = loader.loadModel("models/ball_" + str(ballIndex) + "_center.egg")
            ball.setScale(self.ballSize, self.ballSize, self.ballSize)

            cs = CollisionSphere(0, 0, 0, 1)
            ballSphere = ball.attachNewNode(CollisionNode('ball_' + str(ballIndex)))
            ballSphere.node().addSolid(cs)
            ballSphere.node().setFromCollideMask(BitMask32.bit(0) | BitMask32.bit(1) | BitMask32.bit(3) | BitMask32.bit(4))
            ballSphere.node().setIntoCollideMask(BitMask32.bit(1))

            ball.reparentTo(ballRoot)
            self.ballRoots.append(ballRoot)
            self.balls.append(ball)            
            self.ballSpheres.append(ballSphere)


            ballGroundRay = CollisionRay()     # Create the ray
            ballGroundRay.setOrigin(0, 0, 0)    # Set its origin
            ballGroundRay.setDirection(self.gravityDirection[0], self.gravityDirection[1], self.gravityDirection[2])  # And its direction
            # Collision solids go in CollisionNode
            # Create and name the node
            ballGroundCol = CollisionNode('ball_ray_' + str(ballIndex))
            ballGroundCol.addSolid(ballGroundRay)  # Add the ray
            ballGroundCol.setFromCollideMask(BitMask32.bit(2))  # Set its bitmasks
            ballGroundCol.setIntoCollideMask(BitMask32.allOff())
            # Attach the node to the ballRoot so that the ray is relative to the ball
            # (it will always be 10 feet over the ball and point down)
            ballGroundColNp = ballRoot.attachNewNode(ballGroundCol)
            self.ballGroundRays.append(ballGroundColNp)

            ballRoot.hide()
            continue

        
        # Finally, we create a CollisionTraverser. CollisionTraversers are what
        # do the job of walking the scene graph and calculating collisions.
        # For a traverser to actually do collisions, you need to call
        # traverser.traverse() on a part of the scene. Fortunately, ShowBase
        # has a task that does this for the entire scene once a frame.  By
        # assigning it to self.cTrav, we designate that this is the one that
        # it should call traverse() on each frame.
        self.cTrav = CollisionTraverser()

        # Collision traversers tell collision handlers about collisions, and then
        # the handler decides what to do with the information. We are using a
        # CollisionHandlerQueue, which simply creates a list of all of the
        # collisions in a given pass. There are more sophisticated handlers like
        # one that sends events and another that tries to keep collided objects
        # apart, but the results are often better with a simple queue
        self.cHandler = CollisionHandlerQueue()
        # Now we add the collision nodes that can create a collision to the
        # traverser. The traverser will compare these to all others nodes in the
        # scene. There is a limit of 32 CollisionNodes per traverser
        # We add the collider, and the handler to use as a pair
        
        #self.cTrav.addCollider(self.ballSphere, self.cHandler)
        for ballSphere in self.ballSpheres:
            self.cTrav.addCollider(ballSphere, self.cHandler)
            continue
        for ballGroundRay in self.ballGroundRays:
            self.cTrav.addCollider(ballGroundRay, self.cHandler)
            continue        
        #self.cTrav.addCollider(self.ballGroundColNp, self.cHandler)

        # Collision traversers have a built in tool to help visualize collisions.
        # Uncomment the next line to see it.
        #self.cTrav.showCollisions(render)

        # This section deals with lighting for the ball. Only the ball was lit
        # because the maze has static lighting pregenerated by the modeler
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor((.55, .55, .55, 1))
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setDirection(LVector3(0, 0, -1))
        directionalLight.setColor((0.375, 0.375, 0.375, 1))
        directionalLight.setSpecularColor((1, 1, 1, 1))

        for ballRoot in self.ballRoots:
            ballRoot.setLight(render.attachNewNode(ambientLight))
            ballRoot.setLight(render.attachNewNode(directionalLight))
            continue

        # This section deals with adding a specular highlight to the ball to make
        # it look shiny.  Normally, this is specified in the .egg file.
        m = Material()
        m.setSpecular((1, 1, 1, 1))
        m.setShininess(96)
        for ball in self.balls:
            ball.setMaterial(m, 1)
            continue


        self.original = False
        if self.original:
            camera.setPosHpr(0, 0, 25, 0, -90, 0)        
            self.maze = loader.loadModel("models/maze")
            self.maze.reparentTo(render)
            self.walls = self.maze.find("**/wall_collide")
            self.walls.node().setIntoCollideMask(BitMask32.bit(0))
            self.walls.show()
            pass



        #planeTriangles, planeNormals = self.planeInfo.getPlaneGeometries()

        self.triNPs = []
        for triangleIndex, triangle in enumerate(planeTriangles):
            #print(triangleIndex)
            #for triangle in triangles:
            #print(triangle)
            tri = CollisionPolygon(Point3(triangle[0][0], triangle[0][1], triangle[0][2]), Point3(triangle[1][0], triangle[1][1], triangle[1][2]), Point3(triangle[2][0], triangle[2][1], triangle[2][2]))
            triNP = render.attachNewNode(CollisionNode('tri_' + str(triangleIndex)))
            triNP.node().setIntoCollideMask(BitMask32.bit(0))
            triNP.node().addSolid(tri)
            self.triNPs.append(triNP)
            #triNP.show()
            continue

        #print(horizontalPlaneTriangles)
        
        for triangleIndex, triangle in enumerate(horizontalPlaneTriangles):
            #print(triangleIndex)
            #for triangle in triangles:
            #print(triangle)
            tri = CollisionPolygon(Point3(triangle[0][0], triangle[0][1], triangle[0][2]), Point3(triangle[1][0], triangle[1][1], triangle[1][2]), Point3(triangle[2][0], triangle[2][1], triangle[2][2]))
            triNP = render.attachNewNode(CollisionNode('ground_' + str(triangleIndex)))
            triNP.node().setIntoCollideMask(BitMask32.bit(2))
            triNP.node().addSolid(tri)
            self.triNPs.append(triNP)
            #triNP.show()
            continue
        
        
        # tri = CollisionPolygon(Point3(-1, 4, -1), Point3(2, 4, -1), Point3(2, 4, 2))    
        # triNP = render.attachNewNode(CollisionNode('tri'))
        # triNP.node().setIntoCollideMask(BitMask32.bit(0))
        # triNP.node().addSolid(tri)
        # triNP.show()
        
        
        #self.planeScene.node().setIntoCollideMask(BitMask32.bit(0))
        # roomRootNP = self.planeScene
        # roomRootNP.flattenLight()
        # mesh = BulletTriangleMesh()
        # polygons = roomRootNP.findAllMatches("**/+GeomNode")

        # # p0 = Point3(-10, 4, -10)
        # # p1 = Point3(-10, 4, 10)
        # # p2 = Point3(10, 4, 10)
        # # p3 = Point3(10, 4, -10)
        # # mesh.addTriangle(p0, p1, p2)
        # # mesh.addTriangle(p1, p2, p3)

        # print(polygons)
        # for polygon in polygons:
        #     geom_node = polygon.node()
        #     #geom_node.reparentTo(self.render)
        #     #print(geom_node.getNumGeoms())
        #     ts = geom_node.getTransform()
        #     #print(ts)
        #     for geom in geom_node.getGeoms():
        #         mesh.addGeom(geom, ts)
        #         continue
        #     continue
        # #self.scene = roomRootNP
        # shape = BulletTriangleMeshShape(mesh, dynamic=False)
        # #shape = BulletPlaneShape(Vec3(0, 0, 1), 1)
        # room = BulletRigidBodyNode('scene')
        # room.addShape(shape)
        # #room.setLinearDamping(0.0)
        # #room.setFriction(0.0)
        # print(shape)
        # room.setDeactivationEnabled(False)
        # roomNP = render.attachNewNode(room)
        # roomNP.setPos(0, 0, 0)
        # roomNP.node().setIntoCollideMask(BitMask32.bit(0))
        # self.world = BulletWorld()
        # self.world.setGravity(Vec3(0, 0, 0))
        # self.world.attachRigidBody(roomNP.node())
        #room.setRestitution(1)

        #self.roomNP = self.scene
        

        self.cueRoot = render.attachNewNode("cueRoot")
        self.cue = loader.loadModel("models/cue_center.egg")
        self.cue.setScale(self.cueLength * 3, self.cueLength * 3, self.cueLength)
        self.cue.reparentTo(self.cueRoot)

        self.cuePos = (10, 0, 0)
        
        self.pickerNode = CollisionNode('mouseRay')
        # Attach that node to the camera since the ray will need to be positioned
        # relative to it
        self.pickerNP = camera.attachNewNode(self.pickerNode)
        # Everything to be picked will use bit 1. This way if we were doing other
        # collision we could separate it
        self.pickerNode.setFromCollideMask(BitMask32.bit(2))
        self.pickerNode.setIntoCollideMask(BitMask32.allOff())        
        self.pickerRay = CollisionRay()  # Make our ray
        # Add it to the collision node
        self.pickerNode.addSolid(self.pickerRay)
        # Register the ray as something that can cause collisions
        self.cTrav.addCollider(self.pickerNP, self.cHandler)        

        self.accept("mouse1", self.hit)  # left-click grabs a piece


        self.holeLength = 0.06
        holePos, holeHpr = self.planeInfo.getHolePos()
        self.holeRoot = render.attachNewNode("holeRoot")
        #self.hole = loader.loadModel("models/hole_horizontal_center.egg")
        self.hole = loader.loadModel("models/hole_color.egg")
        #self.hole = loader.loadModel("models/billiards_hole_center.egg")
        self.hole.setScale(self.holeLength, self.holeLength, self.holeLength)
        self.hole.reparentTo(self.holeRoot)
        self.hole.setTwoSided(True)
        self.holeRoot.setPos(holePos[0], holePos[1], holePos[2])
        self.holeRoot.setHpr(holeHpr[0], holeHpr[1], holeHpr[2])
        #tex = loader.loadTexture('models/Black_Hole.jpg')
        #self.hole.setTexture(tex, 1)
        self.holeRoot.hide()
        
        ct = CollisionTube(0, 0, 0, 0, 0.001, 0, 0.5)
        self.holeTube = self.hole.attachNewNode(CollisionNode('hole'))
        self.holeTube.node().addSolid(ct)
        self.holeTube.node().setFromCollideMask(BitMask32.allOff())        
        self.holeTube.node().setIntoCollideMask(BitMask32.bit(4))
        #self.holeTube.show()


        inPortalPos, inPortalHpr, outPortalPos, outPortalHpr, self.portalNormal = self.planeInfo.getPortalPos()
        self.portalLength = 0.06
        self.inPortalRoot = render.attachNewNode("inPortalRoot")
        self.inPortal = loader.loadModel("models/portal_2_center.egg")
        self.inPortal.setScale(self.portalLength, self.portalLength, self.portalLength)
        self.inPortal.reparentTo(self.inPortalRoot)
        self.inPortalRoot.setPos(inPortalPos[0], inPortalPos[1], inPortalPos[2])
        self.inPortalRoot.setHpr(inPortalHpr[0], inPortalHpr[1], inPortalHpr[2])
        self.inPortalRoot.hide()
        
        ct = CollisionTube(0, 0, 0, 0, 0.001, 0, 1)
        self.inPortalTube = self.inPortal.attachNewNode(CollisionNode('portal_in'))
        self.inPortalTube.node().addSolid(ct)
        self.inPortalTube.node().setFromCollideMask(BitMask32.allOff())        
        self.inPortalTube.node().setIntoCollideMask(BitMask32.bit(3))
        #self.inPortalTube.hide()

        self.outPortalRoot = render.attachNewNode("outPortalRoot")
        self.outPortal = loader.loadModel("models/portal_2_center.egg")
        self.outPortal.setScale(self.portalLength, self.portalLength, self.portalLength)
        self.outPortal.reparentTo(self.outPortalRoot)
        self.outPortalRoot.setPos(outPortalPos[0], outPortalPos[1], outPortalPos[2])
        self.outPortalRoot.setHpr(outPortalHpr[0], outPortalHpr[1], outPortalHpr[2])
        self.outPortalRoot.hide()
        
        ct = CollisionTube(0, 0, 0, 0, 0.001, 0, 1)
        self.outPortalTube = self.outPortal.attachNewNode(CollisionNode('portal_out'))
        self.outPortalTube.node().addSolid(ct)
        self.outPortalTube.node().setFromCollideMask(BitMask32.allOff())        
        self.outPortalTube.node().setIntoCollideMask(BitMask32.bit(3))
        #self.outPortalTube.hide()        
        #self.inPortalTube.show()
        #self.outPortalTube.show()
        #self.holeTube.show()
        
        #self.cTrav.addCollider(self.holeTube, self.cHandler)

        background_image = loader.loadTexture('dump/' + str(self.sceneIndex) + '_image.png')
        cm = CardMaker('background')
        cm.setHas3dUvs(True)
        info = np.load('dump/' + str(self.sceneIndex) + '_info.npy')
        #self.camera = getCameraFromInfo(self.info)
        depth = 10.0
        sizeU = info[2] / info[0] * depth
        sizeV = info[6] / info[5] * depth
        cm.setFrame(Point3(-sizeU, depth, -sizeV), Point3(sizeU, depth, -sizeV), Point3(sizeU, depth, sizeV), Point3(-sizeU, depth, sizeV))
        self.card = self.render.attachNewNode(cm.generate())
        self.card.setTransparency(True)    
        self.card.setTexture(background_image)
        self.card.hide()
        
        
        self.ballGroundMap = {}
        self.ballBouncing = np.full(len(self.balls), 3)
        
        self.started = False
        self.start()
        
        self.hitIndex = 0
        
        self.showing = 'none'
        self.showingProgress = 0
        
        partsScene = PartsScene(self.sceneIndex)        
        self.planeNPs, self.planeCenters = partsScene.generateEggModel()
        return

    def start(self):
        #startPos = self.maze.find("**/start").getPos()
        #self.ballRoot.setPos(0.5, 0, 0)
        #self.ballV = LVector3(0, 0.5, 0)         # Initial velocity is 0
        #self.accelV = LVector3(0, 0, 0)        # Initial acceleration is 0

        self.ballVs = []
        self.accelVs = []
        for ballIndex in xrange(len(self.balls)):
            self.ballVs.append(LVector3(0, 0, 0))
            self.accelVs.append(LVector3(0, 0, 0))
            continue
        self.ballRoots[0].setPos(0.2, 1.05, -0.1)
        #self.ballVs[0] = LVector3(0, 0.0, 0)                
        self.ballRoots[1].setPos(0.32, 1.2, -0.1)
        #self.ballRoots[2].setHpr(0, 0, 90)
        self.ballRoots[2].setPos(-0.4, 1.1, 0.4)
        axis = LVector3.up()
        prevRot = LRotationf(self.balls[2].getQuat())
        newRot = LRotationf(axis, 90)
        self.balls[2].setQuat(prevRot * newRot)
            
        # Create the movement task, but first make sure it is not already
        # running
        taskMgr.remove("rollTask")
        #taskMgr.remove("mouseTask")
        self.mainLoop = taskMgr.add(self.rollTask, "rollTask")        
        #self.mainLoop = taskMgr.add(self.mouseTask, "mouseTask")
        
        return

    def hit(self):
        if self.cuePos[0] < 5:
            cueDirection = self.ballRoots[0].getPos() - LVector3(self.cuePos[0], self.cuePos[1], self.cuePos[2])
            #power = cueDirection.length()
            cueDirection = cueDirection / cueDirection.length()
            if self.hitIndex < 0:
                self.ballVs[0] = cueDirection * self.cuePower * 8
            elif self.hitIndex == 0:
                self.ballVs[0] = LVector3(0.5, 0.47, 0)
                self.hitIndex += 1
            elif self.hitIndex == 1:
                self.ballVs[0] = LVector3(0.072, 0.62, 0)
                self.hitIndex += 1
            elif self.hitIndex == 2:
                self.ballVs[0] = LVector3(0.7, 0.0, 0)
                self.hitIndex += 1                                
                pass
            self.started = True
            print('hit', cueDirection)
            self.ballBouncing = np.full(len(self.balls), 3)
            pass



    # This function handles the collision between the ball and a wall
    def planeCollideHandler(self, colEntry):
        #return
        ballName = colEntry.getFromNode().getName()
        ballIndex = int(ballName[5:])
        
        # First we calculate some numbers we need to do a reflection
        # print(colEntry)
        # name = colEntry.getIntoNode().getName()
        # triangleIndex = int(name[4:])
        # print(triangleIndex)
        # print(self.planeNormals[triangleIndex])
        # print(colEntry.getSurfaceNormal(render))
        # exit(1)
        norm = colEntry.getSurfaceNormal(render) * -1  # The normal of the wall
        norm.normalize()
        curSpeed = self.ballVs[ballIndex].length()                # The current speed
        inVec = self.ballVs[ballIndex] / curSpeed                 # The direction of travel
        velAngle = norm.dot(inVec)                    # Angle of incidance
        hitDir = colEntry.getSurfacePoint(render) - self.ballRoots[ballIndex].getPos()
        hitDir.normalize()
        # The angle between the ball and the normal
        hitAngle = norm.dot(hitDir)

        # Ignore the collision if the ball is either moving away from the wall
        # already (so that we don't accidentally send it back into the wall)
        # and ignore it if the collision isn't dead-on (to avoid getting caught on
        # corners)
        #print(velAngle, hitAngle)

        if velAngle > 0 and hitAngle > .995:
            print('plane', ballName, velAngle)
            # Standard reflection equation
            reflectVec = (norm * norm.dot(inVec * -1) * 2) + inVec

            # This makes the velocity half of what it was if the hit was dead-on
            # and nearly exactly what it was if this is a glancing blow
            #self.ballVs[ballIndex] = reflectVec * (curSpeed * (((1 - velAngle) * .5) + .5))
            self.ballVs[ballIndex] = reflectVec * curSpeed
            # Since we have a collision, the ball is already a little bit buried in
            # the wall. This calculates a vector needed to move it so that it is
            # exactly touching the wall
            disp = (colEntry.getSurfacePoint(render) -
                    colEntry.getInteriorPoint(render))
            newPos = self.ballRoots[ballIndex].getPos() + disp
            self.ballRoots[ballIndex].setPos(newPos)
            pass
        return    

    # This function handles the collision between the ball and a wall
    def portal(self, colEntry):
        ballName = colEntry.getFromNode().getName()
        print('portal', ballName)
        ballIndex = int(ballName[5:])
        
        #norm = colEntry.getSurfaceNormal(render) * -1  # The normal of the wall
        norm = LVector3(self.portalNormal[0], self.portalNormal[1], self.portalNormal[2])
        norm.normalize()
        curSpeed = self.ballVs[ballIndex].length()                # The current speed
        inVec = self.ballVs[ballIndex] / curSpeed                 # The direction of travel
        velAngle = norm.dot(inVec)                    # Angle of incidance
        hitDir = colEntry.getSurfacePoint(render) - self.ballRoots[ballIndex].getPos()
        hitDir.normalize()
        # The angle between the ball and the normal
        #print(colEntry.getSurfacePoint(render), self.ballRoots[ballIndex].getPos())
        #print(norm, hitDir)
        hitAngle = norm.dot(hitDir)
        
        # Ignore the collision if the ball is either moving away from the wall
        # already (so that we don't accidentally send it back into the wall)
        # and ignore it if the collision isn't dead-on (to avoid getting caught on
        # corners)
        #print(velAngle, hitAngle)
        #print(velAngle, hitAngle)
        if velAngle > 0:
            print(colEntry.getIntoNode().getName())
            if '_in' in colEntry.getIntoNode().getName():
                self.ballRoots[ballIndex].setPos(self.outPortalRoot.getPos())
            else:
                self.ballRoots[ballIndex].setPos(self.inPortalRoot.getPos())
                pass
            print(self.ballVs[ballIndex], ((norm * norm.dot(inVec * -1) * 2) + inVec) * curSpeed, norm)
            #exit(1)
            self.ballVs[ballIndex] = ((norm * norm.dot(inVec * -1) * 2) + inVec) * curSpeed
            #self.ballVs[ballIndex] *= -1
            pass
        return    
    

    # This function handles the collision between the ball and a wall
    def ballCollideHandler(self, colEntry):
        # First we calculate some numbers we need to do a reflection
        fromBallName = colEntry.getFromNode().getName()
        fromBallIndex = int(fromBallName[5:])
        #if fromBallIndex != 0:
        #return
        intoBallName = colEntry.getIntoNode().getName()
        intoBallIndex = int(intoBallName[5:])        

        print('ball', fromBallName, intoBallName)
        
        norm = colEntry.getSurfaceNormal(render) * -1  # The normal of the wall
        norm = norm / norm.length()
        curSpeed = self.ballVs[fromBallIndex].length()                # The current speed
        inVec = self.ballVs[fromBallIndex] / curSpeed                 # The direction of travel
        velAngle = norm.dot(inVec)                    # Angle of incidance
        hitDir = colEntry.getSurfacePoint(render) - self.ballRoots[fromBallIndex].getPos()
        hitDir.normalize()
        # The angle between the ball and the normal
        hitAngle = norm.dot(hitDir)

        # print(norm)
        # print(self.ballVs[fromBallIndex])
        # print(velAngle, hitAngle)
        # print(self.ballRoots[fromBallIndex].getPos())
        # print(self.ballRoots[intoBallIndex].getPos())        
        # exit(1)
        #print(fromBallIndex, intoBallIndex)
        #exit(1)
        
        # Ignore the collision if the ball is either moving away from the wall
        # already (so that we don't accidentally send it back into the wall)
        # and ignore it if the collision isn't dead-on (to avoid getting caught on
        # corners)
        #print(velAngle, hitAngle)
        if velAngle > 0 and hitAngle > .995:
            # Standard reflection equation
            self.ballVs[fromBallIndex] = ((norm * norm.dot(inVec * -1)) + inVec) * curSpeed

            disp = (colEntry.getSurfacePoint(render) -
                    colEntry.getInteriorPoint(render))
            newPos = self.ballRoots[fromBallIndex].getPos() + disp
            self.ballRoots[fromBallIndex].setPos(newPos)

            self.ballVs[intoBallIndex] = norm * norm.dot(inVec) * curSpeed
            pass
        return    


        
    def groundCollideHandler(self, colEntry):
        # Set the ball to the appropriate Z value for it to be exactly on the
        # ground
        ballName = colEntry.getFromNode().getName()

        if 'mouseRay' in ballName:
            for v in self.ballVs:
                if v.length() > 1e-4:
                    self.cuePos = (10, 0, 0)
                    return
                continue
            #print(self.mouseWatcherNode.hasMouse())
            norm = colEntry.getSurfaceNormal(render)
            norm.normalize()
            touchPoint = colEntry.getSurfacePoint(render)
            cuePoint = touchPoint + norm * self.ballSize
            cueDirection = self.ballRoots[0].getPos() - cuePoint
            self.cuePower = cueDirection.length()
            cueDirection = cueDirection / cueDirection.length()
            cuePoint = self.ballRoots[0].getPos() - cueDirection * self.cueLength
            self.cuePos = cuePoint
            #self.cueRoot.setH(np.rad2deg(np.arctan2(cueDirection[1], cueDirection[0])) + 90)
            self.cueRoot.setH(np.rad2deg(np.arctan2(cueDirection[1], cueDirection[0])) + 90)  
            self.cueRoot.setP(-np.rad2deg(np.arcsin(cueDirection[2])) + 90)
            #self.cueRoot.setP(90)
            #print(np.rad2deg(np.arctan2(cueDirection[1], cueDirection[0])), np.rad2deg(np.arcsin(cueDirection[2])))

            # prevRot = LRotationf(self.cue.getQuat())
            # axis = LVector3.up().cross(self.ballVs[ballIndex])
            # newRot = LRotationf(axis, 45.5 * dt * self.ballVs[ballIndex].length())
            # self.balls[ballIndex].setQuat(prevRot * newRot)
            return
            
        #print('ground', ballName)
        #print(ballName, colEntry.getIntoNode().getName())
        #print(colEntry.getFromNode().getBitMask(), colEntry.getIntoNode().getBitMask())
        ballIndex = int(ballName[9:])

        groundName = colEntry.getIntoNode().getName()
        groundIndex = int(groundName[7:])
        #print(groundIndex)
        #print(self.ballGroundMap)
        if ballIndex == 0 and False:
            print(groundIndex, self.ballGroundMap)
            pass
        
        if ballIndex not in self.ballGroundMap or self.ballGroundMap[ballIndex][0] != groundIndex:
            return
        
        norm = -colEntry.getSurfaceNormal(render)
        norm = norm / norm.length()

        curSpeed = self.ballVs[ballIndex].length()                # The current speed
        inVec = self.ballVs[ballIndex] / max(curSpeed, 1e-4)                 # The direction of travel
        velAngle = norm.dot(inVec)                    # Angle of incidance
        hitDir = colEntry.getSurfacePoint(render) - self.ballRoots[ballIndex].getPos()
        hitDir.normalize()
        # The angle between the ball and the normal
        hitAngle = norm.dot(hitDir)
        
        surfacePos = colEntry.getSurfacePoint(render)
        ballPos = self.ballRoots[ballIndex].getPos()
        surfacePos = ballPos + norm * norm.dot(surfacePos - ballPos)

        distance = norm.dot(surfacePos - ballPos)
        if distance < 0:
            return

        
        if distance < self.ballSize + 1e-3:
            self.ballRoots[ballIndex].setPos(surfacePos - norm * self.ballSize)
            if self.ballVs[ballIndex].length() > 1e-2:
                self.ballVs[ballIndex] = (-norm * velAngle + inVec) * curSpeed
                #self.ballVs[ballIndex] = -norm.cross(norm.cross(self.ballVs[ballIndex]))
                self.accelVs[ballIndex] = -self.ballVs[ballIndex] / self.ballVs[ballIndex].length() * 0.0025
            else:
                self.ballVs[ballIndex] = LVector3(0, 0, 0)
                self.accelVs[ballIndex] = LVector3(0, 0, 0)                
                pass
        else:
            self.accelVs[ballIndex] = self.accelVs[ballIndex] - norm * norm.dot(self.accelVs[ballIndex]) + norm * 0.05
            pass
        return

    
        # if self.started:
        #     if abs(distance - self.ballSize) > 0.001 and abs(distance - self.ballSize) < self.ballSize:
        #         self.ballRoots[ballIndex].setPos(surfacePos - norm * self.ballSize)
        #         pass
        #     self.ballVs[ballIndex] = -norm.cross(norm.cross(self.ballVs[ballIndex]))
        #     if self.ballVs[ballIndex].length() > 1e-3:
        #         self.accelVs[ballIndex] = -self.ballVs[ballIndex] / self.ballVs[ballIndex].length() * 0.015
        #     else:
        #         self.ballVs[ballIndex] = LVector3(0, 0, 0)
        #         self.accelVs[ballIndex] = LVector3(0, 0, 0)
        #         pass
        #     #print(self.ballVs[ballIndex], self.accelVs[ballIndex])
        #     #print(surfacePos - norm * self.ballSize)

        #     return


        if ballIndex == 0:
            print('distance_1', self.started, distance, velAngle, self.ballVs[ballIndex], self.accelVs[ballIndex])
        
        if distance < self.ballSize:
            self.ballRoots[ballIndex].setPos(surfacePos - norm * self.ballSize)
            if velAngle > 0 and hitAngle > .995:
                if abs(velAngle * curSpeed) < 0.2:
                    if ((-norm * velAngle + inVec) * curSpeed).length() < 0.02:
                        self.ballVs[ballIndex] = LVector3(0, 0, 0)
                        self.accelVs[ballIndex] = LVector3(0, 0, 0)
                        pass
                    pass
                else:
                    if self.ballBouncing[ballIndex] > 0:
                        if ballIndex == 0:
                            print('bouncing')
                            pass
                        self.ballVs[ballIndex] = (-norm * velAngle + inVec) * curSpeed * 0.5 - norm * velAngle * curSpeed * 0.25
                        self.accelVs[ballIndex] = LVector3(0, 0, 0)
                        self.ballBouncing[ballIndex] -= 1
                    else:
                        self.ballVs[ballIndex] = (-norm * velAngle + inVec) * curSpeed
                        self.accelVs[ballIndex] = LVector3(0, 0, 0)
                        pass
                    pass
                pass
            
            pass

        if (distance - self.ballSize) > 0.001:
            self.accelVs[ballIndex] = self.accelVs[ballIndex] - norm * norm.dot(self.accelVs[ballIndex]) + norm * 0.1
            # print(self.accelVs[ballIndex] - norm * norm.dot(self.accelVs[ballIndex]))
            # print(norm)
            # print(inVec)
            # print(velAngle)
            # print(-norm * velAngle + inVec)
            # print(norm * 0.01)
            # exit(1)
        elif distance - self.ballSize > -0.001:
            if self.ballVs[ballIndex].length() < 0.001:
                #print('stop', self.ballVs[ballIndex], self.accelVs[ballIndex])

                self.ballVs[ballIndex] = LVector3(0, 0, 0)
                self.accelVs[ballIndex] = LVector3(0, 0, 0)
                self.started = False
            else:
                if abs(velAngle) < 1e-3:
                    self.ballVs[ballIndex] = (-norm * velAngle + inVec) * curSpeed
                    #self.ballVs[ballIndex] = -norm.cross(norm.cross(self.ballVs[ballIndex]))
                    self.accelVs[ballIndex] = -self.ballVs[ballIndex] / self.ballVs[ballIndex].length() * 0.01
                    #print('speed', self.ballVs[ballIndex], self.accelVs[ballIndex])
                    pass
                pass
            pass
    
        # #print(distance - self.ballSize)
        # if (distance - self.ballSize) > 0.01:
        #     self.accelVs[ballIndex] = self.accelVs[ballIndex] - norm * norm.dot(self.accelVs[ballIndex]) + norm * 0.01
        #     #if ballIndex == 0:
        #     #print(velAngle, self.ballVs[ballIndex], self.accelVs[ballIndex], norm)
        #     #pass

        #     print('fall', self.accelVs[ballIndex], distance)
        #     # print(self.accelVs[ballIndex] - norm * norm.dot(self.accelVs[ballIndex]))
        #     # print(norm)
        #     # print(inVec)
        #     # print(velAngle)
        #     # print(-norm * velAngle + inVec)
        #     # print(norm * 0.01)
        #     # exit(1)
        # else:
        #     #hitAngle > .995
        #     #print(velAngle)
        #     #print(norm)

        #     #self.ballRoots[ballIndex].setPos(surfacePos - norm * self.ballSize)
        #     if curSpeed > 1e-1:
        #         print('angle', velAngle, norm)
        #         self.norm = norm
        #         pass
        #     if velAngle > 1e-3:
        #         if curSpeed < 1e-3:
        #             self.ballVs[ballIndex] = LVector3(0, 0, 0)
        #             self.accelVs[ballIndex] = LVector3(0, 0, 0)
        #             self.ballRoots[ballIndex].setPos(surfacePos - norm * self.ballSize)
        #         else:
        #             self.ballVs[ballIndex] = (-norm * velAngle + inVec) * curSpeed * 0.9 - norm * velAngle * curSpeed * 0.25
        #             self.accelVs[ballIndex] = LVector3(0, 0, 0)
        #             pass
        #         #print((-norm * velAngle + inVec) * curSpeed * 0.9, norm * velAngle * curSpeed * 0.25)
        #         #print(curSpeed, norm.dot(self.ballVs[ballIndex]) / self.ballVs[ballIndex].length(), self.ballVs[ballIndex], self.accelVs[ballIndex])
        #         #print(norm.dot(self.ballVs[ballIndex]) / self.ballVs[ballIndex].length(), norm.dot(self.accelVs[ballIndex]) / self.accelVs[ballIndex].length(), self.ballVs[ballIndex], self.accelVs[ballIndex])
        #     elif velAngle > -1e-3:
        #         if self.ballVs[ballIndex].length() < 0.001:
        #             #self.ballVs[ballIndex] = LVector3(0, 0, 0)
        #             #self.accelVs[ballIndex] = LVector3(0, 0, 0)
        #             #print('stop', self.ballVs[ballIndex], self.accelVs[ballIndex])
        #             pass
        #         else:
        #             #self.ballVs[ballIndex] = (-norm * velAngle + inVec) * curSpeed * 0.9
        #             #print(self.ballVs[ballIndex], self.accelVs[ballIndex])
        #             self.accelVs[ballIndex] = -(-norm * velAngle + inVec) * 0.1
        #             print('accel', self.accelVs[ballIndex])
        #             pass
        #         pass
        #     else:
        #         #print('stop', self.ballVs[ballIndex], self.accelVs[ballIndex])
        #         #self.ballVs[ballIndex] = LVector3(0, 0, 0)
        #         #self.accelVs[ballIndex] = LVector3(0, 0, 0)
        #         pass
        #     pass
        return
    
    # This is the task that deals with making everything interactive
    def rollTask(self, task):
        # Standard technique for finding the amount of time since the last
        # frame
        dt = globalClock.getDt()

        # If dt is large, then there has been a # hiccup that could cause the ball
        # to leave the field if this functions runs, so ignore the frame
        if dt > .2:
            return Task.cont

        # if base.mouseWatcherNode.is_button_down('a'):
        #     self.holeRoot.setH(self.holeRoot.getH() + 1)
        #     print(self.holeRoot.getHpr())
        #     pass
        # if base.mouseWatcherNode.is_button_down('s'):
        #     self.holeRoot.setP(self.holeRoot.getP() + 1)
        #     print(self.holeRoot.getHpr())
        #     pass
        # if base.mouseWatcherNode.is_button_down('d'):
        #     self.holeRoot.setR(self.holeRoot.getR() + 1)
        #     print(self.holeRoot.getHpr())
        #     pass
        
        if base.mouseWatcherNode.is_button_down('space') and self.showing == 'none':
            self.showing = 'parts'
            #self.showingProgress = 0
            pass
        #print(self.showing)
        #print(self.showing)
        if self.showing == 'none':
            return Task.cont
        if self.showing == 'parts':
            self.showingProgress += 0.01
            #self.showingProgress += 1
            print(self.showingProgress)
            scale = 2 - self.showingProgress
            scaleY = 1 + (scale - 1) * 0.5
            for planeIndex, planeNP in enumerate(self.planeNPs):
                center = self.planeCenters[planeIndex]
                planeNP.setPos(center[0] * scale, center[1] * scaleY, center[2] * scale)
                planeNP.reparentTo(self.render)
                planeNP.setTwoSided(True)
                continue
            if self.showingProgress > 1:
                self.showing = 'moving'
                for planeIndex, planeNP in enumerate(self.planeNPs):
                    planeNP.removeNode()
                    continue
                self.planeScene.show()
                self.showingProgress = 0
            return Task.cont
        if self.showing == 'moving':
            self.showingProgress += 0.005
            #self.showingProgress += 1
            #print(self.showingProgress, np.sign(self.showingProgress - 0.5) * min(self.showingProgress % 0.5, 0.5 - self.showingProgress % 0.5) * 4)
            self.camera.setPos(np.sign(self.showingProgress - 0.5) * min(self.showingProgress % 0.5, 0.5 - self.showingProgress % 0.5) * 3, 0, 0)
            #self.camera.setHpr(angleDegrees, 0, 0)
            #self.camera.lookAt(0, 0, 0)
            self.camera.lookAt(0, 3, 0)
            if self.showingProgress > 1:
                self.showing = 'geometry'
                self.camera.setPos(0, 0, 0)
                #self.planeScene.removeNode()
                # for triNP in self.triNPs:
                #     triNP.show()
                #     continue
                self.showingProgress = 1
            return Task.cont
        if self.showing == 'geometry':
            self.showingProgress += 0.02
            if self.showingProgress > 1:
                #self.showing = 'image'
                self.showing = 'placement'
                self.showingProgress = 0
                self.holeRoot.show()
                self.inPortalRoot.show()
                self.outPortalRoot.show()
                self.inPortalTube.show()
                self.outPortalTube.show()
                for ballRoot in self.ballRoots:
                    ballRoot.show()
                    continue
                self.showingProgress = 0                
                pass
            return Task.cont
        # if self.showing == 'placement':
        #     self.showingProgress += 0.005
        #         continue

        if self.mouseWatcherNode.hasMouse():
            mpos = self.mouseWatcherNode.getMouse()
            self.mpos = mpos
            self.pickerRay.setFromLens(self.camNode, mpos.getX(), mpos.getY())
            pass
        
        if base.mouseWatcherNode.is_button_down('space') and self.showing == 'placement':
            self.card.show()
            self.planeScene.removeNode()
            self.showing = 'image'
            pass
        # if base.mouseWatcherNode.is_button_down('space') and self.showing == 'image':
        #     for triNP in self.triNPs:
        #         triNP.hide()
        #         continue
        #     self.showing = 'start'
        #     pass
            
        
        # The collision handler collects the collisions. We dispatch which function
        # to handle the collision based on the name of what was collided into


        self.ballGroundMap = {}
        for i in range(self.cHandler.getNumEntries()):
            entry = self.cHandler.getEntry(i)
            ballName = entry.getFromNode().getName()
            groundName = entry.getIntoNode().getName()            
            if 'ball_ray_' not in ballName:
                continue
            if 'ground_' not in groundName:
                continue
            ballIndex = int(ballName[9:])
            groundIndex = int(groundName[7:])
            norm = -entry.getSurfaceNormal(render)
            if norm.length() == 0:
                continue
            norm = norm / norm.length()            
            distance = norm.dot(entry.getSurfacePoint(render) - self.ballRoots[ballIndex].getPos())
            #print(distance)
            if distance < 0:
                continue
            if ballIndex not in self.ballGroundMap or distance < self.ballGroundMap[ballIndex][1]:
                self.ballGroundMap[ballIndex] = (groundIndex, distance)
                pass
            continue
        
        for i in range(self.cHandler.getNumEntries()):
            entry = self.cHandler.getEntry(i)
            fromName = entry.getFromNode().getName()
            #if 'mouseRay' in fromName:
            #continue
            name = entry.getIntoNode().getName()            
            #if name == "plane_collide":
            if 'tri_' in name:
                self.planeCollideHandler(entry)
            #elif name == "wall_collide":
            #self.wallCollideHandler(entry)
            #elif name == "ground_collide":
            #self.groundCollideHandler(entry)
            elif 'ball_' in name:
                self.ballCollideHandler(entry)
            elif 'ground_' in name:
                self.groundCollideHandler(entry)
            elif 'hole' in name:
                self.score(entry)
            elif 'portal_' in name:
                self.portal(entry)
                pass
            continue

        # Read the mouse position and tilt the maze accordingly
        if base.mouseWatcherNode.hasMouse():
            mpos = base.mouseWatcherNode.getMouse()  # get the mouse position
            #self.maze.setP(mpos.getY() * -10)
            #self.maze.setR(mpos.getX() * 10)
            pass

        # if base.mouseWatcherNode.is_button_down('mouse1'):
        #     print(base.mouseWatcherNode.getMouseX())
        #     print(base.mouseWatcherNode.getMouseY())            
        #     exit(1)
            
        # Finally, we move the ball
        # Update the velocity based on acceleration
        for ballIndex in xrange(len(self.balls)):
            if self.ballVs[ballIndex].length() < 1e-4 and self.ballVs[ballIndex].dot(self.accelVs[ballIndex]) < -1e-4:
                self.ballVs[ballIndex] = LVector3(0, 0, 0)
                self.accelVs[ballIndex] = LVector3(0, 0, 0)
            else:
                self.ballVs[ballIndex] += self.accelVs[ballIndex] * dt * ACCEL
                pass
            #print('current speed', self.ballVs[ballIndex], self.accelVs[ballIndex])
            # Clamp the velocity to the maximum speed
            if self.ballVs[ballIndex].lengthSquared() > MAX_SPEED_SQ:
                self.ballVs[ballIndex].normalize()
                self.ballVs[ballIndex] *= MAX_SPEED
                pass
            #print(self.ballVs[ballIndex], self.accelVs[ballIndex], self.ballRoots[ballIndex].getPos())
            
            # Update the position based on the velocity
            self.ballRoots[ballIndex].setPos(self.ballRoots[ballIndex].getPos() + (self.ballVs[ballIndex] * dt))

            # This block of code rotates the ball. It uses something called a quaternion
            # to rotate the ball around an arbitrary axis. That axis perpendicular to
            # the balls rotation, and the amount has to do with the size of the ball
            # This is multiplied on the previous rotation to incrimentally turn it.
            prevRot = LRotationf(self.balls[ballIndex].getQuat())
            axis = LVector3.up().cross(self.ballVs[ballIndex])
            newRot = LRotationf(axis, np.rad2deg(dt * self.ballVs[ballIndex].length() / self.ballSize))
            self.balls[ballIndex].setQuat(prevRot * newRot)
            continue

        self.cueRoot.setPos(self.cuePos[0], self.cuePos[1], self.cuePos[2])
        return Task.cont       # Continue the task indefinitely

    def score(self, colEntry):
        ballName = colEntry.getFromNode().getName()
        if 'ball_' not in ballName:
            return
        print('score', ballName)
        ballIndex = int(ballName[5:])
        self.ballRoots[ballIndex].removeNode()

        del self.ballRoots[ballIndex]
        del self.balls[ballIndex]
        del self.ballSpheres[ballIndex]
        del self.ballGroundRays[ballIndex]        
        del self.ballVs[ballIndex]
        del self.accelVs[ballIndex]
        for otherIndex in xrange(ballIndex, len(self.balls)):
            self.ballSpheres[otherIndex].setName('ball_' + str(otherIndex))
            self.ballGroundRays[otherIndex].setName('ball_ray_' + str(otherIndex))
            continue
        return
        
    # If the ball hits a hole trigger, then it should fall in the hole.
    # This is faked rather than dealing with the actual physics of it.
    def loseGame(self, entry):
        # The triggers are set up so that the center of the ball should move to the
        # collision point to be in the hole
        toPos = entry.getInteriorPoint(render)
        taskMgr.remove('rollTask')  # Stop the maze task

        # Move the ball into the hole over a short sequence of time. Then wait a
        # second and call start to reset the game
        Sequence(
            Parallel(
                LerpFunc(self.ballRoot.setX, fromData=self.ballRoot.getX(),
                         toData=toPos.getX(), duration=.1),
                LerpFunc(self.ballRoot.setY, fromData=self.ballRoot.getY(),
                         toData=toPos.getY(), duration=.1),
                LerpFunc(self.ballRoot.setZ, fromData=self.ballRoot.getZ(),
                         toData=self.ballRoot.getZ() - .9, duration=.2)),
            Wait(1),
            Func(self.start)).start()

# Finally, create an instance of our class and start 3d rendering
demo = BallInMazeDemo()
demo.run()
    

            
