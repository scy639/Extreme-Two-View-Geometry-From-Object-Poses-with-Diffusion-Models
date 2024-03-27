import root_config
import functools
import glob
from pathlib import Path
import cv2
import numpy as np
import os
from pose_util import  *
from skimage.io import imread, imsave
import pickle
import json,math
import os.path as osp
import random, os
import numpy as np
from PIL import Image, ImageFile
import mask_util
from logging_util import ddd
class BaseDatabase:
    @staticmethod
    def _imgFullPaths_2_img_ids__A(imgFullPaths,check=True,SUFFIX='.png'):
        """
        img_id= int(img name)
        """
        ret=[]
        for imgFullPath in imgFullPaths:
            name=os.path.basename(imgFullPath)
            assert name.endswith(SUFFIX),imgFullPath
            assert name.count(SUFFIX)==1
            name:str=name[:-len(SUFFIX)]
            if name.isdigit():
                img_id= int(name)
                ret.append( img_id  )
            else:
                if '_Q0Sipr' in name :
                    pass
                else:
                    assert '_warp' in name,str(imgFullPath)
                    assert name.replace('_warp','').isdigit()
        #sort ret
        ret.sort()
        if check:
            assert ret[-1]==len(ret)-1
        return ret
    def get_K(self, img_id):
        pass
    def get_pose(self, img_id):
        """
        opencv 3,4
        """
        pass
    def get_img_ids(self):
        pass
    def get_image_full_path(self, img_id):
        pass
    def get_mask_full_path(self, img_id):
        pass

    def get_image(self, img_id):
        image_full_path = self.get_image_full_path(img_id)
        img = imread(image_full_path)
        return img
    def get_mask_hw0_255(self, img_id):
        # return np.sum(imread(self.get_mask_full_path(img_id)), -1) > 0
        mask_full_path = self.get_mask_full_path(img_id)
        mask = imread(mask_full_path)
        if(mask.ndim==3):
            mask=mask_util.Mask.hw3__2__hw0(mask)
        assert len(mask.shape)==2 
        assert mask.dtype==np.uint8#0-255
        return mask
    def get_whiteBg_maskedImage(self, img_id):
        img = self.get_image(img_id)
        mask = self.get_mask_hw0_255(img_id)
        masked_image = mask_util.Mask.get_whiteBg_maskedImage_from_hw0_255(img, mask)
        if masked_image.shape[2]!=3:
            assert 0
            # rgba 2 rgb
            masked_image = masked_image[:, :, :3]
            return masked_image
        return masked_image

    def get_bbox(self,img_id):
        mask_hw0_255=self.get_mask_hw0_255(img_id)
        mask_hw0_bool=mask_util.Mask.hw_255__2__hw_bool(mask_hw0_255,THRES=125)
        bbox = mask_util.Mask.mask_hw0_bool__2__bbox(np.array(mask_hw0_bool))
        return bbox
    @staticmethod
    @functools.cache
    def refId2refDatabase(refIdWhenNormal):
        from dataset.database import parse_database_name
        database_name = refIdWhenNormal
        database_type_and_name: str = f"zero123/{database_name}"
        ref_database = parse_database_name(database_type_and_name)
        return ref_database
    
    def get_data_4_finetune(self,img_id0,img_id1=None):
        """
        1. get q0,q1's K,w2c,pathbbox;
        2. warp q0,q1 (get new K,w2c,img and bbox
        3. get q0's o2c
        4. rela_w2c=; q1's o2c= rela_w2c @ q0's o2c
        return q1,K1,q1's o2c
        """
        from infer_pair import look_at_wrapper_wrapper,get_z_4_normObj,pose_compose

        if self.__class__.__name__=='Co3dv2Database':
            seq=self.seq
            cate=self.cate
        else:
            seq=""
            cate=self.obj
        if img_id1==None:
            
            _len=len(self.get_img_ids())
            N=10
            img_id1 = random.randint(0, _len - 1)
            while abs(img_id1-img_id0)<N:
                img_id1 = random.randint(0, _len - 1)
        # 1
        # image0_path = self.get_image_full_path(img_id0)
        # image1_path = self.get_image_full_path(img_id1)
        whiteBg_maskedImage0 = self.get_whiteBg_maskedImage(img_id0)
        whiteBg_maskedImage1 = self.get_whiteBg_maskedImage(img_id1)
        K0 = self.get_K(img_id0)
        K1 = self.get_K(img_id1)
        w2c0 = self.get_pose(img_id0)
        w2c1 = self.get_pose(img_id1)
        w2c0=Pose_R_t_Converter.pose34_2_pose44(w2c0)
        w2c1=Pose_R_t_Converter.pose34_2_pose44(w2c1)
        bbox0 = self.get_bbox(img_id0)
        bbox1 = self.get_bbox(img_id1)
        # 2.
        img0_warp,  pose0_rect,K0 = look_at_wrapper_wrapper(whiteBg_maskedImage0, bbox0, K0)
        img1_warp,  pose1_rect,K1 = look_at_wrapper_wrapper(whiteBg_maskedImage1, bbox1, K1)
        w2c0 = pose_compose(w2c0[:3], pose0_rect)
        w2c1 = pose_compose(w2c1[:3], pose1_rect)
        w2c0=Pose_R_t_Converter.pose34_2_pose44(w2c0)
        w2c1=Pose_R_t_Converter.pose34_2_pose44(w2c1)
        # 3. o2c0
        def _o2c0():
            refIdWhenNormal = root_config.RefIdWhenNormal.get_id(cate, seq, f"+{img_id0}")
            ref_database=self.refId2refDatabase(refIdWhenNormal)
            zero123Input_info = ref_database.get_zero123Input_info()
            o2c0=np.array(zero123Input_info["pose"])
            # z
            assert o2c0[2,3]==0
            img_w=img0_warp.shape[1]
            img_h=img0_warp.shape[0]
            obj_width = img_w
            obj_height = img_h
            z4normObj = get_z_4_normObj(fx=K0[0][0], fy=K0[1][1],
                                        obj_w_pixel=obj_width, obj_h_pixel=obj_height,
                                        img_w=img_w, img_h=img_h)
            o2c0[2, 3]=z4normObj
            return o2c0,ref_database,
        o2c0,ref_database=_o2c0()
        # 4.
        def check_whether_camera_looks_at_origin(extrinsic):
            z=extrinsic[2,3]
            x=extrinsic[0,3]
            y=extrinsic[1,3]
            hori_component=math.sqrt(x**2+y**2)
            assert z>hori_component
        
        # check_whether_camera_looks_at_origin(w2c1)
        check_whether_camera_looks_at_origin(o2c0)
        # w2c0[2,3]=1
        # w2c1[2,3]=1
        # o2c0[2,3]=1
        rela_w2c = w2c1 @  np.linalg.inv(w2c0)
        o2c1 = rela_w2c @ o2c0
        o2c1[2,3]=1
        o2c1[0,3]=0
        o2c1[1,3]=0
        return ref_database,img_id1,img1_warp, K1, o2c1[:3,:]

