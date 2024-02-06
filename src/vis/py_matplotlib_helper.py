"""
from https://github.com/mstranne/py_matplotlib_helper/3D_pose_plot.py
"""
import math

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

"""
    plot an coordinate system to visualize Pose (R|t)
    
    ax      : matplotlib axes to plot on
    R       : Rotation as roation matrix
    t       : translation as np.array (1, 3)
    scale   : Scale as np.array (1, 3)
    l_width : linewidth of axis
    text    : Text written at origin
"""
def plotPose(ax, R, t, scale=np.array((1.0, 0.5, 2.0)), l_width=2, text=None,color_or_3color=("red","green","blue")):
    if(isinstance(color_or_3color,str)):
        xyz_color=[color_or_3color]*3
    else:
        xyz_color=color_or_3color
    SCALE=3
    new_scale =scale*SCALE 

    x_axis = np.array(([0, 0, 0], [1, 0, 0])) * new_scale
    y_axis = np.array(([0, 0, 0], [0, 1, 0])) * new_scale
    z_axis = np.array(([0, 0, 0], [0, 0, 1])) * new_scale

    x_axis += t
    y_axis += t
    z_axis += t

    x_axis = x_axis @ R
    y_axis = y_axis @ R
    z_axis = z_axis @ R

    ax.plot3D(x_axis[:, 0], x_axis[:, 1], x_axis[:, 2], color=xyz_color[0], linewidth=l_width)
    ax.plot3D(y_axis[:, 0], y_axis[:, 1], y_axis[:, 2], color=xyz_color[1], linewidth=l_width)
    ax.plot3D(z_axis[:, 0], z_axis[:, 1], z_axis[:, 2], color=xyz_color[2], linewidth=l_width)

    if (text is not None):
        ax.text(x_axis[0, 0], x_axis[0, 1], x_axis[0, 2], "red")

    return None
#
# def plotPose_B(ax, R, t, scale=np.array((1, 1, 1),dtype=np.float64), l_width=2, text=None,color="black"):
#     x_axis = np.array(([0, 0, 0], [1, 0, 0])) * scale
#     y_axis = np.array(([0, 0, 0], [0, 1, 0])) * scale
#     z_axis = np.array(([0, 0, 0], [0, 0, 1])) * scale
#
#     x_axis += t
#     y_axis += t
#     z_axis += t
#
#     x_axis = x_axis @ R
#     y_axis = y_axis @ R
#     z_axis = z_axis @ R
#
#     X_AXIS_LENGTH=1.3
#     Y_AXIS_LENGTH=0.7
#     Z_AXIS_LENGTH=2
#
#     x_axis*=X_AXIS_LENGTH
#     y_axis*=Y_AXIS_LENGTH
#     z_axis*=Z_AXIS_LENGTH
#
#     ax.plot3D(x_axis[:, 0], x_axis[:, 1], x_axis[:, 2], color=color, linewidth=l_width)
#     ax.plot3D(y_axis[:, 0], y_axis[:, 1], y_axis[:, 2], color=color, linewidth=l_width)
#     ax.plot3D(z_axis[:, 0], z_axis[:, 1], z_axis[:, 2], color=color, linewidth=l_width)
#
#     if (text is not None):
#         ax.text(x_axis[0, 0], x_axis[0, 1], x_axis[0, 2], "red")

    return None
def interpolate(p_from, p_to, num):
    direction = (p_to - p_from) / np.linalg.norm(p_to - p_from)
    distance = np.linalg.norm(p_to - p_from) / (num - 1)

    ret_vec = []

    for i in range(0, num):
        ret_vec.append(p_from + direction * distance * i)

    return np.array(ret_vec)


"""
    plot image (plane) in 3D with given Pose (R|t) of corner point

    ax      : matplotlib axes to plot on
    R       : Rotation as roation matrix
    t       : translation as np.array (1, 3), left down corner of image in real world coord
    size    : Size as np.array (1, 2), size of image plane in real world
    img_scale: Scale to bring down image, since this solution needs 1 face for every pixel it will become very slow on big images 
"""
def plotImage(ax, img, R, t, size=np.array((1, 1)), img_scale=8):
    img_size = (np.array((img.shape[0], img.shape[1])) / img_scale).astype('int32')
    img = cv.resize(img, ((img_size[1], img_size[0])))

    corners = np.array(([0., 0, 0], [0, size[0], 0],
                        [size[1], 0, 0], [size[1], size[0], 0]))

    corners += t
    corners = corners @ R
    xx = np.zeros((img_size[0], img_size[1]))
    yy = np.zeros((img_size[0], img_size[1]))
    zz = np.zeros((img_size[0], img_size[1]))
    l1 = interpolate(corners[0], corners[2], img_size[0])
    xx[:, 0] = l1[:, 0]
    yy[:, 0] = l1[:, 1]
    zz[:, 0] = l1[:, 2]
    l1 = interpolate(corners[1], corners[3], img_size[0])
    xx[:, img_size[1] - 1] = l1[:, 0]
    yy[:, img_size[1] - 1] = l1[:, 1]
    zz[:, img_size[1] - 1] = l1[:, 2]

    for idx in range(0, img_size[0]):
        p_from = np.array((xx[idx, 0], yy[idx, 0], zz[idx, 0]))
        p_to = np.array((xx[idx, img_size[1] - 1], yy[idx, img_size[1] - 1], zz[idx, img_size[1] - 1]))
        l1 = interpolate(p_from, p_to, img_size[1])
        xx[idx, :] = l1[:, 0]
        yy[idx, :] = l1[:, 1]
        zz[idx, :] = l1[:, 2]

    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=img / 255, shade=False)
    return None


def get_test_img(size=np.array((640, 480)), col=True):
    if col:
        img = np.zeros((size[0], size[1], 3))
        for idx in range(0, size[0]):
            for jdx in range(0, int(size[0] / 10)):
                img[idx, int(size[0] / 4) + jdx, 1] = 255
        for idx in range(0, size[1]):
            for jdx in range(0, int(size[1] / 10)):
                img[int(size[0] / 4) + jdx, idx, 2] = 255
        return img
    else:
        img = np.zeros((size[0], size[1]))
        for idx in range(0, size[0]):
            for jdx in range(0, int(size[0] / 10)):
                img[idx, int(size[1] / 4) + jdx] = 255
        for idx in range(0, size[1]):
            for jdx in range(0, int(size[1] / 10)):
                img[int(size[0] / 4) + jdx, idx] = 155
        return img
if(__name__ == '__main__'):

    # testcase = 1  # plotPose
    testcase = 2  # plotImagePlane

    if (testcase == 1):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim3d([-2, 2])
        ax.set_ylim3d([-2, 2])
        ax.set_zlim3d([-2, 2])

        R = np.eye(3)
        t = np.zeros((1, 3))
        scale = np.array(([0.5, 0.5, 0.5]))

        plotPose(ax, R, t, scale)

        t = np.array(([1, 1, 1]))
        plotPose(ax, R, t, scale)

        R_rad = np.array((45.0, 0.0, 45.0)) * math.pi / 180
        R = cv.Rodrigues(R_rad)[0]
        t = np.array(([1, 0.5, 0]))
        plotPose(ax, R, t, scale, l_width=3, text="pose 3")

        plt.show()
    if (testcase == 2):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim3d([-2, 2])
        ax.set_ylim3d([-2, 2])
        ax.set_zlim3d([-2, 2])

        R = np.eye(3)
        t = np.zeros((1, 3))
        img = get_test_img()
        # img = cv.cvtColor(img, cv.COLOR_RGB2BGRA)
        plotImage(ax, img, R, t, size=np.array((1, img.shape[0] / img.shape[1])))

        R_rad = np.array((45.0, 0.0, 45.0)) * math.pi / 180
        R = cv.Rodrigues(R_rad)[0]
        t = np.array(([1, 0.5, 0]))
        plotImage(ax, img, R, t, size=np.array((1, img.shape[0] / img.shape[1])))

        plt.show()
