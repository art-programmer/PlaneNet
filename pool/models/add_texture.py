import csv
import sys
import numpy as np
import cv2

texture = cv2.imread(sys.argv[3])

with open(sys.argv[1]) as modelFile, open(sys.argv[2], 'w') as outputFile:
    modelLoader = csv.reader(modelFile, delimiter=' ')
    xs = []
    ys = []
    zs = []        
    for lineIndex, line in enumerate(modelLoader):
        if len(line) == 0:
            outputFile.write('\n')
            continue
        if line[0] == 'vt':
            continue
        if line[0] == 'v':
            line = line[:4]
            u = max(min(int(round((1 + float(line[1])) * 0.5 * texture.shape[1])), texture.shape[1] - 1), 0)
            v = max(min(int(round((1 + float(line[2])) * 0.5 * texture.shape[0])), texture.shape[0] - 1), 0)
            print(u, v)
            color = texture[v][u].astype(np.float32) / 255
            line.append(str(color[2]))
            line.append(str(color[1]))
            line.append(str(color[0]))
            pass
        outputFile.write(' '.join(line) + '\n')
        continue
    modelFile.close()
    outputFile.close()
    pass
