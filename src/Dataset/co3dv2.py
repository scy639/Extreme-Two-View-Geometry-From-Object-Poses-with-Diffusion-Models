import functools

from imports import *
if __name__ == "__main__":
    from linemod import LinemodDataset
    from BaseDatabase import  BaseDatabase
else:
    from .linemod import LinemodDataset
    from .BaseDatabase import  BaseDatabase
from torchvision import transforms
import glob
import mask_util
from pathlib import Path
import cv2
import numpy as np
import os
from skimage.io import imread, imsave
import pickle
import root_config
import gzip
import json
import os.path as osp
import random, os
import torch
from PIL import Image, ImageFile
from pytorch3d.renderer import PerspectiveCameras
# from torch.utils.data import Dataset
from Dataset.BaseDataset import  BaseDataset
from torchvision import transforms
from logging_util import ddd

# ----------------------------------------------------

# CO3DV2_ROOT = "data/co3d_v2"
# CO3DV2_ROOT = "../preprocess/data/co3d_v2"
# CO3D_PREPROCESSED_ANNOTATION_DIR = "../preprocess/data/co3d_v2_annotations"
# CO3DV2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", CO3DV2_ROOT)
# CO3D_PREPROCESSED_ANNOTATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", CO3D_PREPROCESSED_ANNOTATION_DIR)
# CO3DV2_ROOT = root_config.dataPath_co3d
# ----------------------------------------------------

TRAINING_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

TEST_CATEGORIES = [
    "ball",
    "book",
    "couch",
    "frisbee",
    "hotdog",
    "kite",
    "remote",
    "sandwich",
    "skateboard",
    "suitcase",
]


def co3dv2_cate_2_l_seq(cate):
    cate_dir = f'{CO3DV2_ROOT}/{cate}'
    
    l_seq = [os.path.basename(path) for path in glob.glob(f'{cate_dir}/*') if os.path.isdir(path) ]
    l_seq.remove("eval_batches")
    l_seq.remove("set_lists")
    
    for seq in l_seq:
        assert seq.replace("_", "").isdigit()
    return l_seq
class Co3dv2Database(BaseDatabase) :
    @staticmethod
    def imgpath2int(rela_or_abs_path:str):
        rela_or_abs_path=rela_or_abs_path.split("/")[-1]
        assert rela_or_abs_path.endswith("png") or rela_or_abs_path.endswith("jpg")
        assert rela_or_abs_path.startswith("frame")
        imgInt = rela_or_abs_path[:-4]
        imgInt = imgInt.replace("frame", "")
        imgInt = int(imgInt)
        return imgInt
    def __init__(self, cate:str, seq:str):
        self.cate = cate  # bench,suitcase,...
        self.seq=seq
        # self._dir = f'{CO3DV2_ROOT}/{cate}/{seq}'
        __cate_dir = f'{CO3DV2_ROOT}/{cate}'
        split_name = root_config.SPLIT
        # annotation
        raw_frame_annotation_file = osp.join(CO3DV2_ROOT, __cate_dir,f"frame_annotations.jgz")#raw
        annotation_file = osp.join(CO3D_PREPROCESSED_ANNOTATION_DIR, f"{cate}_{split_name}.jgz")#preprocessed
        with gzip.open(annotation_file, "r") as fin:
            annotation = json.loads(fin.read())
            seq_data = annotation[seq]
            assert len(seq_data) >= 2
            filtered_data = []
            bad_seq = False
            for data in seq_data:
                # Make sure translation31s are not ridiculous
                if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
                    bad_seq = True
                    break
                # Ignore all unnecessary information.
                filtered_data.append(data)
            assert not bad_seq
        # dic: imageFullpath,maskFullpath,size,K,bbox,pose
        i2dic={}
        for dic in filtered_data:
            filepath,bbox,R,T,focal_length,principal_point=dic["filepath"],dic["bbox"],dic["R"],dic["T"],dic["focal_length"],dic["principal_point"]
            imgInt=self.imgpath2int(filepath)
            R=np.array(R)
            R=R.T
            T=np.array(T)
            pose34_pytorch3d=Pose_R_t_Converter.R_t3np__2__pose34(R,T)
            pose34_opencv=pytorch3d_2_opencv__leftMulW2cpose(Pose_R_t_Converter.pose34_2_pose44(pose34_pytorch3d))
            pose34_opencv=pose34_opencv[:3,:]
            i2dic[imgInt]={
                "imageFullpath":os.path.join(CO3DV2_ROOT, filepath),
                "bbox":bbox,
                "pose":pose34_opencv,
            }

        with gzip.open(raw_frame_annotation_file, "r") as fin:
            raw_annotation = json.loads(fin.read())
            """
            [
                {
                    "sequence_name": "12_90_489",
                    "frame_number": 0,
                    "frame_timestamp": -1.0,
                    "image": {
                        "path": "apple/12_90_489/images/frame000001.jpg",
                        "size": [
                            1899,
                            1068
                        ]
                    },
                    "depth": {
                        "path": "apple/12_90_489/depths/frame000001.jpg.geometric.png",
                        "scale_adjustment": 3.1105196475982666,
                        "mask_path": "apple/12_90_489/depth_masks/frame000001.png"
                    },
                    "mask": {
                        "path": "apple/12_90_489/masks/frame000001.png",
                        "mass": 185638.0
                    },
                    "viewpoint": {
                        "R": [[
                                0.5353747606277466,
                                0.7598133087158203,
                                0.36885982751846313
                            ], .., .. ],
                        "T": [
                            2.355828285217285,
                            1.9350138902664185,
                            17.140409469604492
                        ],
                        "focal_length": [
                            3.554843402533942,
                            1.9992484222781728
                        ],
                        "principal_point": [
                            0.0,
                            0.0
                        ]
                    }
                },...]
            """
            for dic in raw_annotation:
                if dic["sequence_name"]!=seq:
                    continue
                image_path,image_size,mask_path=dic["image"]["path"],dic["image"]["size"],dic["mask"]["path"]
                imgInt=self.imgpath2int(image_path)
                # fx=dic["viewpoint"]["focal_length"][0]
                # fy=dic["viewpoint"]["focal_length"][1]
                f = np.sqrt(image_size[0] ** 2 + image_size[1] ** 2)
                fx,fy=f,f
                i2dic[imgInt]={
                    **i2dic[imgInt],
                    "maskFullpath":os.path.join(CO3DV2_ROOT, mask_path),
                    "size":image_size,
                    "K":np.array([[fx,0,image_size[0]/2],
                                  [0,fy,image_size[1]/2],
                                  [0,0,1]]),
                }
        for i in range(len(filtered_data)):
            if i not in i2dic:
                #find nearest_dic
                nearest_dic=None
                for j in range(1,99):
                    if i-j in i2dic:
                        nearest_dic=i2dic[i-j]
                        break
                    if i+j in i2dic:
                        nearest_dic=i2dic[i+j]
                        break
                assert nearest_dic is not None
                i2dic[i]=nearest_dic
                ddd(f"frame{i}.xxx 不存在，使用nearest_dic来充替")
        i2dic_=[i2dic[i] for i in i2dic]
        imageFullpaths=[dic["imageFullpath"] for dic in i2dic_]
        maskFullpaths=[dic["maskFullpath"] for dic in i2dic_]
        Ks=[dic['K'] for dic in i2dic_]
        poses=[dic['pose'] for dic in i2dic_]
        bboxs=[dic['bbox'] for dic in i2dic_]

        self.poses, self.Ks ,self.bboxs,self.imageFullpaths,self.maskFullpaths= poses, Ks,bboxs,imageFullpaths,maskFullpaths
        self._img_ids = list(range(len(i2dic_)))

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
            debug_imsave(root_config.path_4debug + f"/{self.__class__.__name__}-{cate}/{seq}-poses.jpg", vis_img)
            l_whiteBg_maskedImage=[self.get_whiteBg_maskedImage(k)  for k in l_key]
            img_num_per_row=int(math.sqrt(len(l_whiteBg_maskedImage)))
            vis_img2=cv2_util.concat_images_list(*l_whiteBg_maskedImage,vert=0,img_num_per_row=img_num_per_row)
            debug_imsave(root_config.path_4debug + f"/{self.__class__.__name__}-{cate}/{seq}-images.jpg", vis_img2)
            print(1)
        # scy_debug_check_pose_in_database()


    def get_K(self, img_id):
        return self.Ks[img_id]
    def get_pose(self, img_id):
        return self.poses[img_id]

    def get_img_ids(self):
        return self._img_ids.copy()

    def get_image_full_path(self, img_id):
        return self.imageFullpaths[img_id]
    def __get_mask_full_path(self, img_id):#might need to resize
        return self.maskFullpaths[img_id]
    def get_mask_hw0_255(self, img_id):
        """
        modify based on relpose++ co3d_v2.py
        """
        image_full_path = self.get_image_full_path(img_id)
        image = Image.open(image_full_path).convert("RGB")
        mask_path = self.__get_mask_full_path(img_id)
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size)
        mask=np.array(mask)
        # mask = mask > 125# then mask.dtype is bool
        if(mask.ndim==3):
            mask=mask_util.Mask.hw3__2__hw0(mask)
        assert len(mask.shape)==2 
        assert mask.dtype==np.uint8#0-255
        return mask
    """       和BaseDatabase的效果几乎一样 
    def get_whiteBg_maskedImage(self, img_id):
        image_full_path = self.get_image_full_path(img_id)
        image = Image.open(image_full_path).convert("RGB")
        white_image = Image.new("RGB", image.size, (255, 255, 255))
        mask_path = self.__get_mask_full_path(img_id)
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size)
        mask = Image.fromarray(np.array(mask) > 125)
        image = Image.composite(image, white_image, mask)
        image=np.array(image)
        return image
        """

class Co3dv2Dataset(LinemodDataset):
    def __init__(
            self,
            category:str ,
    ):
        self.cate=category
        self.sequence_list=co3dv2_cate_2_l_seq(cate=category)
    @functools.cache
    def seq_2_database(self,seq):
        database=Co3dv2Database(cate=self.cate,seq=seq)
        return database
    def __len__(self):
        raise NotImplementedError
    def seq_2_l_index(self,seq):
        return self.seq_2_database(seq).get_img_ids()
    def get_seq_img_ids(self,seq):
        database=self.seq_2_database(seq)
        return database.get_img_ids()
    def get_data(self, sequence_name, index0,index1, q0ipr=0,q1ipr=0, ):
        assert q0ipr==0
        assert q1ipr==0
        self.database=self.seq_2_database(sequence_name)
        ret= super().get_data(sequence_name,index0,index1)
        self.database=None
        return ret
    def get_data_4gen6d(self, sequence_name=None, ids=(0, 1) ,q0ipr=0, q1ipr=0,):
        assert q0ipr==0
        assert q1ipr==0
        self.database=self.seq_2_database(sequence_name)
        ret=super().get_data_4gen6d(sequence_name,ids)
        self.database=None
        return ret
