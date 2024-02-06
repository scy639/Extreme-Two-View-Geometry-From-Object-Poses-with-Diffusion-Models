import os
import cv2
from skimage.io import imsave

# from util_4_e2vg import ImagePathUtil
# from util_4_e2vg import CameraMatrixUtil
# from util_4_e2vg.IntermediateResult import IntermediateResult


# import ImagePathUtil
# import CameraMatrixUtil
# from IntermediateResult import IntermediateResult


def get_xyzLinearGradient(xyzStops: tuple, N: int):
    """
    param:
        xyzStops:((x=0,y=0,z=0,stop=0),(x=0,y=90,z=0,stop=0.2),(x=80,y=90,z=1,stop=0.3),...(,,,1.0))
        N:xyzLinearGradient长度
    return:
        xyzLinearGradient:[(x=0,y=0,z=0),(x=,y=,z=),...]
    """
    xyzLinearGradient = []

    
    num_stops = len(xyzStops)

    
    for i in range(N):
        stop = i / (N - 1)

        
        for j in range(num_stops - 1):
            if stop >= xyzStops[j][3] and stop <= xyzStops[j + 1][3]:
                
                t = (stop - xyzStops[j][3]) / (xyzStops[j + 1][3] - xyzStops[j][3])
                x = xyzStops[j][0] + (xyzStops[j + 1][0] - xyzStops[j][0]) * t
                y = xyzStops[j][1] + (xyzStops[j + 1][1] - xyzStops[j][1]) * t
                z = xyzStops[j][2] + (xyzStops[j + 1][2] - xyzStops[j][2]) * t
                xyzLinearGradient.append((x, y, z))
                break

    return xyzLinearGradient
class OutputIm_Name_Parser:
    @staticmethod
    def parse_A(folder):
        # return  i2samples
        files = os.listdir(folder)
        i2samples = {}
        for file in files:
            # if is jpg
            if (file.split('.')[-1] != 'jpg'):
                continue
            # file=11-2(x=0,y=30.0,z=0).png=i-j(x=0,y=30.0,z=0).png
            i = int(file.split('-')[0])
            j = int(file.split('-')[1].split('(')[0])  # index of sample
            rest = file[len(f"{i}-j"):]
            if (i not in i2samples):
                i2samples[i] = []
            i2samples[i].append(file)
        return  i2samples
    @staticmethod
    def parse_B(folder,in_fullPath=True):
        i2samples=OutputIm_Name_Parser.parse_A(folder)
        ret=[]
        for i in range(len(i2samples)):
            samples=i2samples[i]
            assert len(samples)==1
            ret.append(samples[0])
        if in_fullPath:
            ret=[os.path.join(folder,i) for i in ret]
        return ret