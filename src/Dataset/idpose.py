import functools
import math
import  mask_util
import root_config
import PIL

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../../../..")))
    sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))
    # print("sys.path[0]:", os.path.abspath(sys.path[0]))
    # print("sys.path:", sys.path)
from imports import *

if __name__ == "__main__":
    from linemod import LinemodDataset
    from BaseDatabase import  BaseDatabase
else:
    from .linemod import LinemodDataset
    from .BaseDatabase import  BaseDatabase
from torchvision import transforms
import glob
from pathlib import Path
import cv2
import numpy as np
import os
import plyfile
from skimage.io import imread, imsave
import pickle
import json
import os.path as osp
import random, os
import numpy as np
import torch
from PIL import Image, ImageFile





class IdposeAboDatabase(BaseDatabase) :
    """
    _img_ids==0,1,....(等差数列)。_img_id ==imgInt
    """
    def __init__(self, obj,DATASET_ROOT = root_config.dataPath_idposeAbo):
        self.obj = obj  # bed001,bed002,....
        self._dir = f'{DATASET_ROOT}/{self.obj}'
        self._img_ids=self._imgFullPaths_2_img_ids__A(glob.glob(f'{self._dir}/images/*.png'))
        assert len(self._img_ids)==len(os.listdir(f'{self._dir}/masks'))
        assert self._img_ids==list(range(len(self._img_ids)))
        # self.object_center = np.zeros(3, dtype=np.float32)
        # self.object_vert = np.asarray([0, 0, 1], np.float32)
        # self.Ks=None
        self._id2imgInt=self.__get_id2imgInt()
        self._img_ids = [k for k in range(len(self._id2imgInt))]
        def scy_debug_check_pose_in_database():
            _INTERVAL = 1
            # if(obj=="bed_002"):
            #     l_key =list(range(26,173,_INTERVAL))
            # else:
            #     l_key=list(range(0,len(self._img_ids),_INTERVAL))
            #     TMP_N=200
            #     if(len(l_key)>TMP_N):
            #         l_key=l_key[:TMP_N]
            l_key = list(range(0, len(self._img_ids), _INTERVAL))
            l_key=[self._img_ids[k]for k in l_key]
            TMP_N = 200
            if (len(l_key) > TMP_N):
                l_key = l_key[:TMP_N]
            l_t=[self.get_pose(k)[:,3] for k in l_key]
            # print("l_t",l_t)
            l_w2c = [np.concatenate([self.get_pose(k), np.array([[0, 0, 0, 1]])], axis=0) for k in l_key]
            l_w2c_i = [np.linalg.inv(w2c) for w2c in l_w2c]#c2w
            l_t2=[w2c_i[:,3] for w2c_i in l_w2c_i]
            l__cameraX_inW=[w2c[0,:3] for w2c in l_w2c]
            l__cameraY_inW=[w2c[1,:3] for w2c in l_w2c]
            l__cameraZ_inW=[w2c[2,:3] for w2c in l_w2c]
            from vis.vis_rel_pose import vis_w2cPoses
            param = dict(
                l_w2c=l_w2c,
                y_is_vertical=0,
            )
            view0 = vis_w2cPoses(**param, no_margin=1, )
            view1 = vis_w2cPoses(**param, no_margin=1, kw_view_init=dict(elev=30, azim=60))
            view2 = vis_w2cPoses(**param, no_margin=1, kw_view_init=dict(elev=15, azim=180))
            view3 = vis_w2cPoses(**param, no_margin=1, kw_view_init=dict(elev=45, azim=240))
            vis_img = cv2_util.concat_images_list(
                view0,
                view1,
                view2,
                view3,
                vert=0
            )
            cv2_util.putText(
                vis_img,
                f"{l_key}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (100, 100, 100),
            )
            debug_imsave(root_config.path_4debug + f"/{self.__class__.__name__}-{obj}/poses(after __get_id2imgInt.jpg", vis_img)
            l_whiteBg_maskedImage=[self.get_whiteBg_maskedImage(k)  for k in l_key]
            img_num_per_row=int(math.sqrt(len(l_whiteBg_maskedImage)))
            vis_img2=cv2_util.concat_images_list(*l_whiteBg_maskedImage,vert=0,img_num_per_row=img_num_per_row)
            debug_imsave(root_config.path_4debug + f"/{self.__class__.__name__}-{obj}/images(after __get_id2imgInt.jpg", vis_img2)
            print(1)

        # scy_debug_check_pose_in_database()

    def get_image_full_path(self, img_id):
        img_id = self._id2imgInt[img_id]
        return f'{self._dir}/images/{int(img_id):03}.png'

    @functools.cache
    def get_K(self, img_id):
        # return np.copy(self.Ks[img_id])
        img=Image.open(self.get_image_full_path(img_id))
        image_size = img.size#w,h
        f = np.sqrt(image_size[0] ** 2 + image_size[1] ** 2)
        fx, fy = f, f
        K=np.array([[fx, 0, image_size[0]/2],
                           [0, fy, image_size[1]/2],
                           [0, 0, 1]])
        return K

    def get_pose(self, img_id):
        if hasattr(self,"_id2imgInt"):
            img_id=self._id2imgInt[img_id]
        c2w44 = np.load(os.path.join(self._dir, 'poses', f'{img_id:03d}.npy'))
        # c2w44=Pose_R_t_Converter.pose34_2_pose44(c2w34)
        w2c44=np.linalg.inv(c2w44)
        #openGL 2 openCV
        w2c44_openCV=opengl_2_opencv__leftMulW2cpose(w2c44)
        w2c34_openCV=w2c44_openCV[:3,:]
        return  w2c34_openCV

    def get_img_ids(self):
        return self._img_ids.copy()

    def get_mask_full_path(self, img_id):
        img_id=self._id2imgInt[img_id]
        return f'{self._dir}/masks/{int(img_id):03}.png'
    def __get_id2imgInt(self):
        id2imgInt = []
        imgInt=0
        def overlook(w2c34):
            val=w2c34[2][2]#w2c34[2] is camera z axis's world coordinate.
            return val<=0 
        while (imgInt < len(self._img_ids)):
            w2c34=self.get_pose(imgInt)
            if overlook(w2c34):
                id2imgInt.append(imgInt)
            imgInt+=1
        ddd(f"[{self.__class__.__name__}]id2imgInt='{id2imgInt}'", )
        return  id2imgInt


class IdposeAboDataset(LinemodDataset):
    def __init__(self, category: str):
        # super().__init__(category)
        self.sequence_list = [""]
        self.database = IdposeAboDatabase(obj=category)

class IdposeOmniDatabase(IdposeAboDatabase) :
    def __init__(self, obj,):
        super().__init__(obj=obj,DATASET_ROOT=root_config.dataPath_idposeOmni)
    def get_image_full_path(self, img_id):
        img_id = self._id2imgInt[img_id]
        png= f'{self._dir}/images/{int(img_id):03}.png'
        jpg= f'{self._dir}/images/{int(img_id):03}.jpg'
        if not os.path.exists(jpg):
            #png rgba 2 rgb and save as jpg
            img=PIL.Image.open(png)
            assert img.mode=="RGBA"
            img=img.convert("RGB")
            img.save(jpg)
        return jpg
    
class IdposeOmniDataset(LinemodDataset):
    def __init__(self, category: str):
        # super().__init__(category)
        self.sequence_list = [""]
        self.database = IdposeOmniDatabase(obj=category, )
