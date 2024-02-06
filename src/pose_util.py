import numpy as np
import os,sys
from PIL import Image
from zero123.zero1.util_4_e2vg import CameraMatrixUtil
from zero123.zero1.util_4_e2vg.CameraMatrixUtil import xyz2pose,get_z_4_normObj
# from gen6d.Gen6D.scy.ElevationUtil import eleRadian_2_baseXyz_lXyz
def R_t_2_pose(R,t):
    if(isinstance(R,list)):
        R=np.array(R)
    if(isinstance(t,list)):
        t=np.array(t)
    if(t.shape==(3,)):
        t=t.reshape((3,1))
    t=t.reshape((3,))
    assert(t.shape==(3,))
    assert(R.shape==(3,3))
    pose=np.zeros((4,4))
    pose[:3,:3]=R
    pose[:3,3]=t
    pose[3,3]=1
    return pose
class Pose_R_t_Converter: 
    @staticmethod
    def pose_2_Rt(pose44_or_pose34):
        assert pose44_or_pose34.shape==(4,4) or pose44_or_pose34.shape==(3,4)
        R = pose44_or_pose34[:3, :3]
        t = pose44_or_pose34[:3, 3]
        return R,t
    @staticmethod
    # def Rt_2_pose44(R,t):
    def R_t3np__2__pose44(R,t):
        assert R.shape==(3,3)
        assert t.shape==(3,)
        pose=np.eye(4)
        pose[:3,:3]=R
        pose[:3,3]=t
        return pose
    @staticmethod
    def R_t3np__2__pose34(R,t):
        assert R.shape==(3,3)
        assert t.shape==(3,)
        pose44=Pose_R_t_Converter.R_t3np__2__pose44(R,t)
        pose34=pose44[:3,:]
        return pose34
    @staticmethod
    def pose34_2_pose44(pose34):
        assert pose34.shape==(3,4)
        pose44=np.concatenate([pose34,np.array([[0,0,0,1]])],axis=0)
        return pose44
    @staticmethod
    def R__2__arbitrary_t_pose44(R):
        assert R.shape==(3,3)
        pose44=Pose_R_t_Converter.R_t3np__2__pose44(R,np.zeros((3,)))
        return pose44

def opencv_2_pytorch3d__leftMulW2cR(R):#w2opencv to w2pytorch3d
    assert R.shape==(3,3)
    Rop = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ], dtype=np.float64)#o means OpenCV, p means pytorch3d
    R = Rop @ R
    return R
def opencv_2_pytorch3d__leftMulW2cpose(pose):#TODO check correctness
    assert pose.shape==(4,4)
    Poseop = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)#o means OpenCV, p means pytorch3d
    pose = Poseop @ pose
    return pose
def opencv_2_pytorch3d__leftMulRelR(R):
    assert R.shape==(3,3)
    Rop = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    R = Rop @ R @ (Rop.T)
    return R
# def pytorch3d_2_opencv__leftMulRelR(R):
#     assert R.shape==(3,3)
#     Rop = np.array([
#         [-1, 0, 0],
#         [0, -1, 0],
#         [0, 0, 1],
#     ], dtype=np.float64)
#     R = Rop @ R @ (Rop.T)
#     return R
def opencv_2_pytorch3d__leftMulRelPose(pose):
    assert pose.shape==(4,4)
    Pop = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    ], dtype=np.float64)
    pose = Pop @ pose @ np.linalg.inv(Pop)
    return pose
def pytorch3d_2_opencv__leftMulRelPose(pose):
    return opencv_2_pytorch3d__leftMulRelPose(pose)
def pytorch3d_2_opencv__leftMulW2cpose(pose):
    assert pose.shape==(4,4)
    pose=opencv_2_pytorch3d__leftMulW2cpose(pose)
    return pose
def opengl_2_opencv__leftMulW2cpose(pose):#TODO check correctness
    assert pose.shape==(4,4)
    Posego = np.array([#GL to OpenCV
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    pose = Posego @ pose
    return pose


def compute_angular_error(rotation1, rotation2):
    # R_rel = rotation1.T @ rotation2
    R_rel =   rotation2 @ rotation1.T
    tr = (np.trace(R_rel) - 1) / 2 
    theta = np.arccos(tr.clip(-1, 1))
    return theta * 180 / np.pi
def compute_translation_error(t31_1, t31_2):
    assert t31_1.shape==(3,1)
    assert t31_2.shape==(3,1)
    # ret=np.linalg.norm(t31_1 - t31_2, axis=1)
    # angle between two vectors
    ret=np.arccos(np.dot(t31_1.T,t31_2)/(np.linalg.norm(t31_1)*np.linalg.norm(t31_2)))
    ret=ret.item()
    ret=ret * 180 / np.pi
    return ret


def in_plane_rotate_camera(degree_clockwise, pilImage: Image.Image, w2opencv_44,fillcolor=(255, 255, 255),):
    """
    相机follows opencv convention
    顺时针旋转相机 degree_clockwise ° <--> 逆时针旋转图片 degree_clockwise °
    degree_clockwise 即草稿纸《12.26 for Q0Sipr 》上的θ
    """
    assert -360 <= degree_clockwise <= 360
    if isinstance(pilImage,np.ndarray):
        pilImage=Image.fromarray(pilImage)
    assert isinstance(pilImage,Image.Image)
    assert w2opencv_44.shape == (4, 4)
    img_rot: Image.Image = pilImage.rotate(
        degree_clockwise, 
        fillcolor=fillcolor,
        resample=Image.BICUBIC,
    )
    rad_clockwise = np.deg2rad(degree_clockwise)
    P_IPR = np.asarray([  
        [np.cos(rad_clockwise), np.sin(rad_clockwise), 0, 0],
        [-np.sin(rad_clockwise), np.cos(rad_clockwise), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], np.float32)
    w2opencv_44_rot = P_IPR @ w2opencv_44
    return img_rot,w2opencv_44_rot
def in_plane_rotate_camera_wrap(degree_clockwise, pilImage_or_ndarray, w2opencv_44=None,fillcolor=(255, 255, 255),):
    """
    相机follows opencv convention
    顺时针旋转相机 degree_clockwise ° <--> 逆时针旋转图片 degree_clockwise °
    degree_clockwise 即草稿纸《12.26 for Q0Sipr 》上的θ
    """
    if w2opencv_44 is None:
        w2opencv_44=np.eye(4)
    if isinstance(pilImage_or_ndarray,np.ndarray):
        pilImage=Image.fromarray(pilImage_or_ndarray)
    img_rot,w2opencv_44_rot=in_plane_rotate_camera(degree_clockwise, pilImage, w2opencv_44,fillcolor=fillcolor)
    if isinstance(pilImage_or_ndarray,np.ndarray):
        img_rot=np.array(img_rot)
    return img_rot,w2opencv_44_rot