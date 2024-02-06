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
import torch
from PIL import Image, ImageFile



class GsoDatabase(BaseDatabase) :
    """
    _img_id ==imgInt.   
    """
    def __init__(self, obj:str,):
        assert obj.startswith("GSO_") or obj.startswith("gso_")
        DATASET_ROOT = root_config.dataPath_gso
        self.obj = obj  # bed001,bed002,....
        self._dir = f'{DATASET_ROOT}/{self.obj}'
        self._img_ids=self._imgFullPaths_2_img_ids__A( glob.glob(f'{self._dir}/*.png'))
        self.poses, self.K = self.__get_poses_K()
        assert len(self.poses)==len(self._img_ids)
        assert self._img_ids==list(range(len(self._img_ids)))
        def scy_debug_check_pose_in_database():
            _INTERVAL = 10
            # if(obj=="bed_002"):
            #     l_key =list(range(26,173,_INTERVAL))
            # else:
            #     l_key=list(range(0,len(self._img_ids),_INTERVAL))
            #     TMP_N=200
            #     if(len(l_key)>TMP_N):
            #         l_key=l_key[:TMP_N]
            l_key = list(range(0, len(self._img_ids), _INTERVAL))
            l_key=[self._img_ids[k]for k in l_key]
            TMP_N = 10
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
            debug_imsave(root_config.path_4debug + f"/{self.__class__.__name__}-{obj}/poses.jpg", vis_img)
            l_whiteBg_maskedImage=[self.get_whiteBg_maskedImage(k)  for k in l_key]
            img_num_per_row=int(math.sqrt(len(l_whiteBg_maskedImage)))
            vis_img2=cv2_util.concat_images_list(*l_whiteBg_maskedImage,vert=0,img_num_per_row=img_num_per_row)
            debug_imsave(root_config.path_4debug + f"/{self.__class__.__name__}-{obj}/images.jpg", vis_img2)
            print(1)

        # scy_debug_check_pose_in_database()

    def __get_poses_K(self):
        """
        pose=[R;t]
        xcam = R @ xw + t
        opencv坐标系
        """
        def read_pickle(pkl_path):
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        K, poses = read_pickle(os.path.join(self._dir,'meta.pkl'))


        return poses, K
    def get_K(self, img_id):
        return self.K
    def get_pose(self, img_id):
        return self.poses[img_id]

    def get_img_ids(self):
        return self._img_ids.copy()

    def _get_rgbaImage_full_path(self,img_id):
        fullpath=f'{self._dir}/{int(img_id):03}.png'
        return fullpath
    def _get_rgbaImage(self,img_id):
        fullpath=self._get_rgbaImage_full_path(img_id)
        img=imread(fullpath)
        return img
    def get_image_full_path(self, img_id):
        png= self._get_rgbaImage_full_path(img_id)
        rgb_folder=Path(f'{self._dir}/scyRGB')
        rgb_folder.mkdir(exist_ok=1)
        jpg= f'{str(rgb_folder)}/{int(img_id):03}.png'
        if not os.path.exists(jpg):
            #png rgba 2 rgb(white bg) and save as jpg
            img=PIL.Image.open(png)
            assert img.mode=="RGBA"
            width = img.width
            height = img.height
            image = Image.new('RGB', size=(width, height), color=(255, 255, 255))
            image.paste(img, (0, 0), mask=img)
            image.save(jpg)
        return jpg
    def get_mask_full_path(self, img_id):
        mask_dir= f'{self._dir}/mask'
        os.makedirs(mask_dir,exist_ok=1)
        mask_fullpath = f'{mask_dir}/{int(img_id):03}.png'
        if not os.path.exists(mask_fullpath):
            rgbaImage=self._get_rgbaImage(img_id)
            mask=mask_util.Mask.rgbaImage__2__hw0_255(rgbaImage)
            imsave(mask_fullpath,mask)
        return mask_fullpath

    






class GsoDataset(LinemodDataset):
    def __init__(self, category: str):
        # super().__init__(category)
        self.sequence_list = [""]
        self.database = GsoDatabase(obj=category, )
