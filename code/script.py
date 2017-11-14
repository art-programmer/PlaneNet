import os

def resultsFigure():
    for image_index in [4, 10, 22, 23, 50, 53, 56, 58, 59, 67, 71, 73, 74, 90, 100, 118, 134, 145]:
        command = 'python predict.py --methods=2 --numOutputPlanes=10 --useCache=1 --startIndex=' + str(int(image_index / 30) * 30) + ' --imageIndex=' + str(image_index) + ' --suffix=final'
        os.system(command)
        continue
    return


def failureCases():
    for image_index in [20, 21]:    
        command = 'python predict.py --methods=2 --numOutputPlanes=10 --useCache=1 --startIndex=' + str(int(image_index / 30) * 30) + ' --imageIndex=' + str(image_index) + ' --suffix=failure'
        os.system(command)
        continue
    return


#resultsFigure()
failureCases()
