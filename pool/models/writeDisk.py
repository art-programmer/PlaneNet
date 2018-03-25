import csv
import sys
import numpy as np
import cv2

texture = cv2.imread(sys.argv[1])
texture = cv2.resize(texture, (256, 256))
textureWidth = texture.shape[1]
textureHeight = texture.shape[0]

numPoints = 0
faces = []
indicesMap = {}
for y in xrange(textureHeight):
    for x in xrange(textureWidth):
        if np.sqrt(pow(x - textureWidth / 2, 2) + pow(y - textureHeight / 2, 2)) < min(textureHeight, textureWidth) / 2:
            indicesMap[y * textureWidth + x] = numPoints
            numPoints += 1
            pass        
        continue
    continue

for y in xrange(textureHeight):
    for x in xrange(textureWidth):
        neighbors = []
        for (u, v) in [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]:
            if np.sqrt(pow(u - textureWidth / 2, 2) + pow(v - textureHeight / 2, 2)) < min(textureHeight, textureWidth) / 2:
                neighbors.append((u, v))
                pass
            continue
        if len(neighbors) == 4:
            face = []
            face.append(indicesMap[neighbors[0][1] * textureWidth + neighbors[0][0]])
            face.append(indicesMap[neighbors[1][1] * textureWidth + neighbors[1][0]])
            face.append(indicesMap[neighbors[2][1] * textureWidth + neighbors[2][0]])                
            faces.append(face)
            face = []
            face.append(indicesMap[neighbors[0][1] * textureWidth + neighbors[0][0]])
            face.append(indicesMap[neighbors[2][1] * textureWidth + neighbors[2][0]])
            face.append(indicesMap[neighbors[3][1] * textureWidth + neighbors[3][0]])
            faces.append(face)
        elif len(neighbors) == 3:
            face = []
            face.append(indicesMap[neighbors[0][1] * textureWidth + neighbors[0][0]])
            face.append(indicesMap[neighbors[1][1] * textureWidth + neighbors[1][0]])
            face.append(indicesMap[neighbors[2][1] * textureWidth + neighbors[2][0]])                
            faces.append(face)
            pass
        continue
    continue

with open(sys.argv[2], 'w') as outputFile:
    header = """ply
format ascii 1.0
element vertex """
    header += str(numPoints)
    header += """
property float x
property float y
property float z
property uchar red                                     { start of vertex color }
property uchar green
property uchar blue
element face """
    header += str(len(faces))
    header += """
property list uchar int vertex_indices
end_header
"""
    outputFile.write(header)
    for y in xrange(textureHeight):
        for x in xrange(textureWidth):
            if np.sqrt(pow(x - textureWidth / 2, 2) + pow(y - textureHeight / 2, 2)) < min(textureHeight, textureWidth) / 2:
                color = texture[y][x]
                outputFile.write(str(float(x) / textureWidth * 2 - 1) + ' ' + str(float(y) / textureHeight * 2 - 1) + ' 0.0 ' + str(color[2]) + ' ' + str(color[1]) + ' ' + str(color[0]) + '\n')
                pass
            continue
        continue
    for face in faces:
        outputFile.write('3 ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + '\n')
        continue
    outputFile.close()
    pass
