import numpy as np
import json, math

"""
对img操作过程中内参外参相关的变换
"""


def get_K(img_h, img_w):
    cx = img_w / 2
    cy = img_h / 2
    f = np.sqrt(img_h ** 2 + img_w ** 2)
    fx = f
    fy = f
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


def xyz2pose(x, y, z):
    """
    相机相对obj极坐标为：x,y,z
    x:Polar angle (vertical rotation in degrees)
    y:Azimuth angle (horizontal rotation in degrees)
    z:Zoom (relative distance from center)

    相机前方为z轴，下方为y轴
    相机对着obj(obj在图像中心
    return: pose:object pose means a translation t and a rotation R that transform the object coordinate xobj to the camera coordinate xcam = R xobj+t
            pose=[R,t;0,1]

    """

    p = math.radians(-x)
    a = math.radians(y)
    R_AC = np.array([  
        [-np.sin(a), np.cos(a), 0],
        [0, 0, -1],
        [-np.cos(a), - np.sin(a), 0],
    ])
    R_CD = np.array([
        [1, 0, 0],
        [0, np.cos(p), -np.sin(p)],
        [0, np.sin(p), np.cos(p)],
    ])  
    R = R_CD @ R_AC
    
    z_m = z
    t = np.array([[0, 0, z_m]]).T
    pose = np.concatenate([R, t], 1)
    # pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], 0)
    return pose


def get_z_4_normObj(fx, fy, obj_w_pixel, obj_h_pixel,img_w,img_h):
    """
    obj_w_pixel: obj width in pixel
    normObj means 物体在单位立方体中(其实是2,2,2 cube

    """
    # if (obj_w_pixel >= obj_h_pixel):
    if (obj_w_pixel/obj_h_pixel >= img_w/img_h):
        z = fx / obj_w_pixel *2
    else:
        z = fy / obj_h_pixel *2
    return z
def get_fx_fy_4_normObj(z,obj_w_pixel,obj_h_pixel,img_w,img_h):
    if (obj_w_pixel/obj_h_pixel >= img_w/img_h):
        fx = z * obj_w_pixel/2
        fy = z * obj_w_pixel/2
    else:
        fx = z * obj_h_pixel/2
        fy = z * obj_h_pixel/2
    return fx,fy

def crop(K, coord0: tuple, coord1: tuple):
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    fx_new = fx
    fy_new = fy
    cx_new = cx - coord0[0]
    cy_new = cy - coord0[1]
    K_new = np.array([[fx_new, 0, cx_new], [0, fy_new, cy_new], [0, 0, 1]])
    return K_new


def resize(K, h_old, w_old, h_new, w_new):
    """
    K_new=S@K,where S=[sx,0,0;0,sy,0;0,0,1],sx=w_new/w_old.
    推导过程见草稿纸(和copilot直接生成的也是一样的
    """
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    fx_new = fx * w_new / w_old
    fy_new = fy * h_new / h_old
    cx_new = cx * w_new / w_old
    cy_new = cy * h_new / h_old
    K_new = np.array([[fx_new, 0, cx_new], [0, fy_new, cy_new], [0, 0, 1]])
    return K_new
