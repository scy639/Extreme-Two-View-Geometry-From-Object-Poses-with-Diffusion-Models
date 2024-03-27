import os.path
import root_config
from . import cv2_util
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pose_util import *

# if (__name__ == '__main__'):
if (__package__ == '' or __package__ == None):
    from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
    from py_matplotlib_helper import  plotPose
else:
    from .extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
    from .py_matplotlib_helper import  plotPose

def look_at_origin_R2t_w2cColumn(R,
                                #  distance=15,#4paper-11.09.19:00 and before
                                 # distance=25,
                                 distance=20,
                                 ):
    """
    distance:camera to origin

    Rt遵循w2cColumn
    1.   camera_center_row_under_w=-R[2], camera_center_column_under_w=camera_center_row_under_w.T
    2. according to: camera_center_column_under_w == -R.T @ t. we can get t=
    3. norm t so that |t|==1
    """
    assert R.shape==(3,3)
    camera_center_row_under_w=-R[2]
    camera_center_column_under_w=camera_center_row_under_w.T
    t=(-R.T).T@camera_center_column_under_w
    t=t/np.linalg.norm(t)
    t*=distance
    return t
def look_at_origin_pose2pose_w2cColumn(pose):
    """
    pose遵循w2cColumn
    """
    assert pose.shape==(4,4)
    R=pose[:3,:3]
    t=pose[:3,3]
    new_t=look_at_origin_R2t_w2cColumn(R)
    new_pose=np.eye(4)
    new_pose[:3,:3]=R
    new_pose[:3,3]=new_t
    return new_pose
def get_rainbow_color(num, keyPoint_color):
    """poe比cursor好用
    :param num: how many colors
    :param keyPoint_color: [[r,g,b],...]
    :return: list of colors
    """
    if num <= 0:
        return []

    if len(keyPoint_color) == 0:
        return []

    if len(keyPoint_color) == 1:
        return [keyPoint_color[0]] * num

    colors = []
    segment_count = len(keyPoint_color) - 1
    segment_length = num // segment_count
    remainder = num % segment_count

    for i in range(segment_count):
        start_color = keyPoint_color[i]
        end_color = keyPoint_color[i+1]
        if(i==segment_count-1):
            segment_length+=remainder
        for j in range(segment_length):
            r = start_color[0] + ((end_color[0] - start_color[0]) * j) / segment_length
            g = start_color[1] + ((end_color[1] - start_color[1]) * j) / segment_length
            b = start_color[2] + ((end_color[2] - start_color[2]) * j) / segment_length
            colors.append([r, g, b])
    # print("rainbow colors:",colors)
    return colors
def get_rainbow_color_B(num,keyPoint_color:list=[]):
    """
    :param num: how many colors
    :param keyPoint_color: [[r,g,b],...]
    :return:
    """
    # assert num>=len(keyPoint_color)
    # COLORS = []
    # for i in range(num):
    #     if(i<len(keyPoint_color)):
    #         COLORS.append(keyPoint_color[i])
    #     else:
    #         COLORS.append([np.random.randint(0, 255) / 255, np.random.randint(0, 255) / 255, np.random.randint(0, 255) / 255])
    # return COLORS




    cm = plt.get_cmap('rainbow')
    num_colors = np.linspace(0, 1, num)
    colors = [cm(x) for x in num_colors]
    return colors
def add_base(w2c,base_w2c):
    """
    :param w2c: 4,4
    :param base_w2c: 4,4
    :return:
    """
    assert w2c.shape==(4,4)
    assert base_w2c.shape==(4,4)
    return w2c@base_w2c



def vis_w2cPoses(l_w2c,l_color=None, base_w2c=np.eye(4),do_not_show_base=False, title="",y_is_vertical=True,no_margin=False,kw_view_init={"elev":None, "azim":None}):
    assert base_w2c.shape == (4, 4)
    """
    w2c = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 20],
            [0, 0, 0, 1],
        ])
    """
    LIM=10
    visualizer = CameraPoseVisualizer([-LIM, LIM], [-LIM, LIM], [-LIM, LIM])
    if(l_color is None):
        # l_color = ['r', 'orange', 'yellow', 'g', 'b', 'purple']
        l_color = get_rainbow_color(len(l_w2c),
                                   # keyPoint_color=[[255 / 255, 0, 0], [0, 1, 0], [85 / 255, 0, 127 / 255]],
                                   keyPoint_color=[ [255/255,0,0], [255/255,165/255,0], [255/255,255/255,0], [0,128/255,0], [0,0,255/255], [128/255,0,128/255] ],
                                   )
    for i, w2c in enumerate(l_w2c):
        assert w2c.shape == (4, 4)
        w2c=add_base(w2c,base_w2c)
        visualizer.w2cColumnExtrinsic_2_pyramid(look_at_origin_pose2pose_w2cColumn(w2c), l_color[i], 10,y_is_vertical=y_is_vertical)
    if(not do_not_show_base):
        visualizer.w2cColumnExtrinsic_2_pyramid(look_at_origin_pose2pose_w2cColumn(base_w2c), 'grey', 10,y_is_vertical=y_is_vertical)
    visualizer.show(title=title,kw_view_init=kw_view_init)
    return visualizer.get_img(no_margin=no_margin)
def vis_w2cRelPoses(w2c, gt_w2c, base_w2c=np.eye(4), title="",y_is_vertical=True):
    return vis_w2cPoses([w2c,gt_w2c],l_color=['b','g'],base_w2c=base_w2c,title=title,y_is_vertical=y_is_vertical)
# def vis_w2cPoses_B(l_w2c, base_w2c=np.eye(4), title="",y_is_vertical=True):
#     assert base_w2c.shape == (4, 4)
#     """
#     like vis_w2cRelPoses
#     """
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     LIM = 10
#     ax.set_xlim3d([-LIM, LIM])
#     ax.set_ylim3d([-LIM, LIM])
#     ax.set_zlim3d([-LIM, LIM])
#     # rainbow
#     COLORS = get_rainbow_color(len(l_w2c),keyPoint_color=[ [255/255,0,0],[0,1,0],[85/255,0,127/255]])
#     for i, w2c in enumerate(l_w2c):
#         new_w2c=look_at_origin_pose2pose_w2cColumn(w2c)
#         R=new_w2c[:3,:3]
#         t=new_w2c[:3,3]
#         plotPose(ax, R, t,color_or_3color=COLORS[i]  )
#         # visualizer.w2cColumnExtrinsic_2_pyramid(, COLORS[i], 10,y_is_vertical=y_is_vertical)
#
#     plt.show()
#     # return visualizer.get_img()

# def vis_w2cRelPoses(w2c, gt_w2c, base_w2c=np.eye(4), title="",y_is_vertical=True):
#     assert w2c.shape == (4, 4)
#     assert gt_w2c.shape == (4, 4)
#     assert base_w2c.shape == (4, 4)
#     """
#     :param w2c:
#         w2c = np.array(
#         [
#             [1, 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 1, 20],
#             [0, 0, 0, 1],
#         ])
#     :param gt_w2c:
#     :param base_w2c:
#     :return:
#     """
#     LIM = 10
#     visualizer = CameraPoseVisualizer([-LIM, LIM], [-LIM, LIM], [-LIM, LIM])

#     # visualizer.extrinsic2pyramid(np.linalg.inv(w2c), 'b', 10)
#     # visualizer.extrinsic2pyramid( np.linalg.inv(gt_w2c), 'g', 10)
#     # visualizer.extrinsic2pyramid(np.linalg.inv(base_w2c), 'grey', 10)
#     visualizer.w2cColumnExtrinsic_2_pyramid(look_at_origin_pose2pose_w2cColumn(w2c), 'b', 10,y_is_vertical=y_is_vertical)
#     # visualizer.w2cColumnExtrinsic_2_pyramid(look_at_origin_pose2pose_w2cColumn(np.linalg.inv(w2c)), 'b', 10,y_is_vertical=y_is_vertical)
#     visualizer.w2cColumnExtrinsic_2_pyramid(  look_at_origin_pose2pose_w2cColumn(gt_w2c),'g', 10,y_is_vertical=y_is_vertical)
#     # visualizer.w2cColumnExtrinsic_2_pyramid(  look_at_origin_pose2pose_w2cColumn(np.linalg.inv(gt_w2c)),'y', 10,y_is_vertical=y_is_vertical)
#     visualizer.w2cColumnExtrinsic_2_pyramid(look_at_origin_pose2pose_w2cColumn(base_w2c), 'grey', 10,y_is_vertical=y_is_vertical)
#     visualizer.show(title=title)
#     return visualizer.get_img()

class PoseVisualizer():
    def __init__(self,camera_style='A',allow_list=False):
        self.camera_style=camera_style
        self.allow_list=allow_list
        self.init()
    def init(self):
        """
        l_pose,l_color,l_opacity,l_size
        """
        self.l_pose=[]
        self.l_note=[]
        self.l_color=[]
        self.l_opacity=[]
        self.l_size=[]
        self.l_colorWho=[]
        LIM = 10#4paper-11.09.19:00 and before
        LIM=15
        self._visualizer = CameraPoseVisualizer([-LIM, LIM], [-LIM, LIM], [-LIM, LIM],camera_style=self.camera_style)
    @property
    def size(self):
        return len(self.l_pose)
    def append(self,pose,color,note="",opacity=0.85,size=1,colorWho='mesh'):
        self.l_pose.append(pose)
        self.l_note.append(note)
        self.l_color.append(color)
        self.l_opacity.append(opacity)
        self.l_size.append(size)
        self.l_colorWho.append(colorWho)
    def append_R(self,R,color,note="",T=None,opacity=0.85,size=1,colorWho='mesh'):
        if self.allow_list:
            if isinstance(R,list):
                R=np.array(R,dtype=np.float64)
        if T is None:
            pose=Pose_R_t_Converter.R__2__arbitrary_t_pose44(R)
            pose=look_at_origin_pose2pose_w2cColumn(pose)
        else:
            pose=Pose_R_t_Converter.R_t3np__2__pose44(R,T)
        self.l_pose.append(pose)
        self.l_note.append(note)
        self.l_color.append(color)
        self.l_opacity.append(opacity)
        self.l_size.append(size)
        self.l_colorWho.append(colorWho)
    def get_img(self,base_w2c=np.eye(4),base_color='grey',do_not_show_base=False, title="",y_is_vertical=True,no_margin=False,kw_view_init={"elev":None, "azim":None},show=False,**kw)->np.ndarray:
        assert base_w2c.shape == (4, 4)
        
        assert len(self.l_pose)==len(self.l_color)==len(self.l_opacity)==len(self.l_size)==len(self.l_colorWho)
        for pose, color, opacity, size,colorWho in zip(self.l_pose, self.l_color, self.l_opacity, self.l_size,self.l_colorWho):
            assert pose.shape == (4, 4)
            pose = add_base(pose, base_w2c)
            self._visualizer.w2cColumnExtrinsic_2_pyramid(
                look_at_origin_pose2pose_w2cColumn(pose), color,
                10 * size, y_is_vertical=y_is_vertical, opacity=opacity,
                colorWho=colorWho,
            )
        if(not do_not_show_base):
            self._visualizer.w2cColumnExtrinsic_2_pyramid(look_at_origin_pose2pose_w2cColumn(base_w2c), base_color, 10,y_is_vertical=y_is_vertical)
        if show:
            # if root_config.FOR_PAPER:
            if 0:
                self._visualizer.customize_legend(['Reference']+self.l_note,[base_color]+self.l_color)
            self._visualizer.show(title=title,kw_view_init=kw_view_init)
        return self._visualizer.get_img(no_margin=no_margin,**kw)
    def get_img__mulView_A(self,vert=False,base_w2c=np.eye(4),do_not_show_base=False, title="",y_is_vertical=True,show=False):
        param = dict(
            do_not_show_base=do_not_show_base,
            title=title,
            base_w2c=base_w2c,
            y_is_vertical=y_is_vertical,
            no_margin=1,
        )
        view0 = self.get_img(**param,show=show)
        view1 = self.get_img(**param, kw_view_init=dict(elev=30, azim=60),show=show)
        view2 = self.get_img(**param, kw_view_init=dict(elev=15, azim=180),show=show)
        view3 = self.get_img(**param, kw_view_init=dict(elev=45, azim=240),show=show)
        ret = cv2_util.concat_images_list(view0, view1, view2, view3, vert=vert)
        return ret
    def get_pose_str_A(self):
        import math
        l=[]
        for pose, color, opacity, size in zip(self.l_pose, self.l_color, self.l_opacity, self.l_size):
            s=f"({color},{opacity},{size:.2f}):z={','.join([f'{fp:.3f}' for fp in pose[2][:3]])}\n    y={[f'{fp:.3f}' for fp in pose[1][:3]]}"
            elev_degree=math.degrees(math.asin(pose[2][2]))
            azim_degree=math.degrees(math.atan2(pose[2][1],pose[2][0]))
            s+=f".elev={elev_degree:.3f},azim={azim_degree:.3f}"
            l.append(s)
        ret= "\n".join(l)
        # print("[get_pose_str_A]",ret)
        return ret
    def clear(self):
        self.init()
# if(__name__ == '__main__'):
#     from skimage.io import imsave, imread
#
#     # pose = np.eye(4)
#     # pose[:3, :3] = np.array(
#     #     [
#     #         [1, 0, 0],
#     #         [0, 1, 0],
#     #         [0, 0, 1],
#     #     ]
#     #     # [
#     #     #     [0, 0, 1],
#     #     #     [0, 1, 0],
#     #     #     [-1, 0, 0],
#     #     # ]
#     # )
#     pose = np.array(
#         [
#             [0, 0, 1, 0],
#             [0, 1, 0, 0],
#             [-1, 0, 0, 20],
#             [0, 0, 0, 1],
#         ]
#     )
#     vis_img=vis_w2cRelPoses(pose,pose@pose)
#     imsave("./vis_w2cRelPoses.jpg",vis_img)

#
#
# if (__name__ == '__main__'):
#     LIM = 10
#     focal_len_scaled = 10
#     visualizer = CameraPoseVisualizer([-LIM, LIM], [-LIM, LIM], [-LIM, LIM])
#     base_w2c = np.eye(4)
#     visualizer.extrinsic2pyramid(base_w2c.T, 'grey', focal_len_scaled)
#     # rainbow
#     COLORS = ['r', 'orange', 'yellow', 'g', 'b', 'purple']
#     from numpy import array, float32
#
#     l_w2c1 = [
#         array([[2.02793066e-01, 9.79220917e-01, 1.16417520e-03,
#                 -2.21608356e-03],
#                [7.04484848e-01, -1.45070277e-01, -6.94734266e-01,
#                 -2.21607715e-03],
#                [-6.80129446e-01, 1.41707437e-01, -7.19265555e-01,
#                 2.98355918e+00]]),
#         array([[0.20089464, 0.9791063, 0.0314983, -0.0073753],
#                [0.65541005, -0.11044181, -0.7471548, -0.07522397],
#                [-0.7280653, 0.1707437, -0.66390336, 3.5028007]],
#               dtype=float32),
#         array([[0.23520868, 0.970393, 0.05490372, -0.02211877],
#                [0.6864872, -0.12587447, -0.7161641, -0.15846284],
#                [-0.6880496, 0.20613869, -0.69576913, 3.587253]],
#               dtype=float32),
#         array([[0.2818913, 0.959015, 0.02876706, -0.01891274],
#                [0.74567556, -0.20011848, -0.6355475, -0.13475566],
#                [-0.6037428, 0.20060617, -0.7715257, 3.581127]],
#               dtype=float32)
#     ]
#     l_w2c2 = [
#         array([[-6.30927448e-01, -7.68593619e-01, -1.05803418e-01,
#                 -1.72564249e-03],
#                [-4.53548327e-01, 4.76030266e-01, -7.53451432e-01,
#                 -1.72568199e-03],
#                [6.29463618e-01, -4.27386243e-01, -6.48934937e-01,
#                 2.98217256e+00]]),
#         array([[-0.6215842, -0.778053, -0.09092086, -0.07609062],
#                [-0.39275485, 0.4099681, -0.8232069, -0.01707302],
#                [0.6777733, -0.47598282, -0.56041384, 3.3203177]],
#               dtype=float32),
#         array([[-0.6353853, -0.76725924, -0.08717036, -0.10741311],
#                [-0.3253582, 0.36838013, -0.87088346, -0.04500639],
#                [0.70030534, -0.524985, -0.4836975, 3.5882134]],
#               dtype=float32),
#         array([[-0.6535364, -0.75097495, -0.09448098, -0.11351153],
#                [-0.31544742, 0.3837104, -0.867905, -0.03707656],
#                [0.68802845, -0.5374038, -0.48766196, 3.6617265]],
#               dtype=float32)
#     ]
#     def arr_append_0001(x):
#         return np.concatenate([x, np.array([[0, 0, 0, 1]])], axis=0)
#     l_w2c1=[arr_append_0001(i) for i in l_w2c1]
#     l_w2c2=[arr_append_0001(i) for i in l_w2c2]
#
#
#     l_w2c=[i.T @ j for i,j in zip(l_w2c1,l_w2c2)]
#     for i, w2c in enumerate(l_w2c):

#         # w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0)
#         visualizer.extrinsic2pyramid(w2c.T, COLORS[i], focal_len_scaled)
#     visualizer.show()

# if (__name__ == '__main__'):
#     colors=get_rainbow_color(10,keyPoint_color=[ [255/255,0,0],[0,1,0],[85/255,0,127/255]])
#     print(colors)
#     input("...")
if (__name__ == '__main__'):
    # pose_save_path_format=rf"C:\Users\YiLucky\Desktop\[gen6d-4]hydrant-106_12648_23157-44,80 {{}}.npy"
    # pose_save_path_format=rf"C:\Users\YiLucky\Desktop\[gen6d-4]hydrant-106_12648_23157-68,55 {{}}.npy"
    # pose_save_path_format=rf"C:\Users\YiLucky\Desktop\[gen6d-4]hydrant-106_12648_23157-78,93 {{}}.npy"
    pose_save_path_format=rf"C:\Users\YiLucky\Desktop\[gen6d-4]hydrant-106_12648_23157-95,65 {{}}.npy"

    w2c=np.load(pose_save_path_format.format("w2c"))
    gt_w2c=np.load(pose_save_path_format.format("gt_w2c"))
    print(w2c,gt_w2c,w2c @ gt_w2c,sep="\n")


    # LIM = 10
    # visualizer = CameraPoseVisualizer([-LIM, LIM], [-LIM, LIM], [-LIM, LIM])
    
    # visualizer.extrinsic2pyramid(gt_w2c, 'g', 10)
    # visualizer.extrinsic2pyramid(np.eye(4), 'grey', 10)
    # visualizer.show(title="")

    Rop=np.array([
        [-1,0,0],
        [0,-1,0],
        [0,0,1],
    ],dtype=np.float64)
    w2c[:3,:3]=Rop@(w2c[:3,:3])@(Rop.T)
    # vis_img=vis_w2cRelPoses(w2c,gt_w2c,y_is_vertical=1)
    vis_img=vis_w2cRelPoses(w2c,gt_w2c,y_is_vertical=1)
    # vis_img=vis_w2cRelPoses(np.linalg.inv(w2c),gt_w2c)
    print(1)
# if (__name__ == '__main__'):
#     pose_save_path=rf"C:\Users\YiLucky\Desktop\Zero123CustomDatabase-leftMulPose.npy"
#     l_pose=np.load(pose_save_path)
#     print("np.load:",l_pose)
#
#     # l_pose=np.concatenate([l_pose,[l_pose[-1]@l_pose[0]]],axis=0)
#     # l_pose = [np.linalg.inv(pose) for pose in l_pose]
#
#     vis_img=vis_w2cPoses(l_pose,y_is_vertical=0)
#     print(1)