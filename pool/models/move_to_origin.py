import csv
import sys
import numpy as np

with open(sys.argv[1]) as modelFile:
    modelLoader = csv.reader(modelFile, delimiter=' ')
    xs = []
    ys = []
    zs = []        
    for lineIndex, line in enumerate(modelLoader):
        if len(line) == 0:
            continue
        if line[0] == 'v':
            xs.append(float(line[1]))
            ys.append(float(line[2]))
            zs.append(float(line[3]))
            pass
        continue
    modelFile.close()
    pass

xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)
print(xs.shape)
minX = xs.min()
maxX = xs.max()
minY = ys.min()
maxY = ys.max()
minZ = zs.min()
maxZ = zs.max()
centerX = (minX + maxX) / 2
centerY = (minY + maxY) / 2
centerZ = (minZ + maxZ) / 2
sizeX = (maxX - minX)
sizeY = (maxY - minY)
sizeZ = (maxZ - minZ)
scale = 2 / max(sizeX, sizeY, sizeZ)

with open(sys.argv[1]) as modelFile, open(sys.argv[2], 'w') as outputFile:
    modelLoader = csv.reader(modelFile, delimiter=' ')
    xs = []
    ys = []
    zs = []        
    for lineIndex, line in enumerate(modelLoader):
        if len(line) == 0:
            outputFile.write('\n')
            continue
        if line[0] == 'v':
            line[1] = str((float(line[1]) - centerX) * scale)
            line[2] = str((float(line[2]) - centerY) * scale)
            line[3] = str((float(line[3]) - centerZ) * scale)
            pass
        outputFile.write(' '.join(line) + '\n')
        continue
    modelFile.close()
    outputFile.close()
    pass
