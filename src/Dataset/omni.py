if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../../../..")))
    sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))
    # print("sys.path[0]:", os.path.abspath(sys.path[0]))
    # print("sys.path:", sys.path)
from imports import *

if __name__ == "__main__":
    from BaseDataset import BaseDataset
    from linemod import LinemodDataset
    from BaseDatabase import BaseDatabase
else:
    from .BaseDataset import BaseDataset
    from .BaseDatabase import BaseDatabase
    from .linemod import LinemodDataset
from image_util import *
from mask_util import *
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


def read_pickle(pkl_path):  # from Gen6d
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):  # from Gen6d
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)




class OMNIDatabase(BaseDatabase):
    """
    NOTE: _img_id !=imgInt.   
            img_id:int(img name) ,ie imgInt. 而不是index of img in folder.
    """
    def __init__(self, obj):
        self.obj = obj  # bed001,bed002,....
        self.__dir = f'{omniVideosProcessed_ROOT}/{self.obj}/standard'
        self._img_ids = self._imgFullPaths_2_img_ids__A(  glob.glob(f'{self.__dir}/images/*.jpg') ,check=False,SUFFIX=".jpg" )
        
        tmp_l_originalIndex:list=self._imgFullPaths_2_img_ids__A(  glob.glob(f'{self.__dir}/matting/*.jpg') ,check=False ,SUFFIX=".jpg")
        self.__img_id_2_originalIndex={k:tmp_l_originalIndex.index(k) for k in self._img_ids}
        #---
        self.poses, self.Ks = self.__get_poses_Ks()
        assert len( glob.glob(f'{self.__dir}/images/*.jpg') ) <= len( glob.glob(f'{self.__dir}/matting/*.jpg') )
        assert len( glob.glob(f'{self.__dir}/matting/*.jpg') ) == len(self.poses)
        # self.object_center = np.zeros(3, dtype=np.float32)
        # self.object_vert = np.asarray([0, 0, 1], np.float32)
        def scy_debug_check_pose_in_database():
            TMP_N = 100
            _INTERVAL =len(self._img_ids)// TMP_N
            # if(obj=="bed_002"):
            #     l_key =list(range(26,173,_INTERVAL))
            # else:
            #     l_key=list(range(0,len(self._img_ids),_INTERVAL))
            #     TMP_N=200
            #     if(len(l_key)>TMP_N):
            #         l_key=l_key[:TMP_N]
            l_key = list(range(0, len(self._img_ids), _INTERVAL))
            l_key=[self._img_ids[k]for k in l_key]
            if (len(l_key) > TMP_N):
                l_key = l_key[:TMP_N]
            l_w2c = [np.concatenate([self.get_pose(k), np.array([[0, 0, 0, 1]])], axis=0) for k in l_key]
            # l_w2c = [np.linalg.inv(w2c) for w2c in l_w2c]
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
            debug_imsave(root_config.path_4debug + f"/OMNIDatabase-poses/{obj}.jpg", vis_img)
            # np.save(root_config.path_4download+"/Zero123CustomDatabase-leftMulPose.npy",l_w2c)
        def  scy_debug__vis_images_B():
            TMP_N = 20
            _INTERVAL =len(self._img_ids)// TMP_N
            l_key = list(range(0, len(self._img_ids), _INTERVAL))
            l_key=[self._img_ids[k]for k in l_key]
            if (len(l_key) > TMP_N):
                l_key = l_key[:TMP_N]
            l=[]
            SIZE=128
            for id_ in l_key:
                maskedImage=self.get_whiteBg_maskedImage(id_)
                if 1:
                    tmp_mask=self.get_mask_hw0_255(id_)
                    tmp_mask=Mask.hw_255__2__hw_bool(tmp_mask)
                    unique_values = np.unique(tmp_mask)
                    if not np.array_equal(unique_values, np.array([0, 1])):
                        tmp_s=f'warning:\nnp.unique={unique_values.tolist()}'
                        print(tmp_s)
                        maskedImage=cv2_util.putText_B(maskedImage,tmp_s,fontScale=2)
                        maskedImage=cv2.resize(maskedImage,(SIZE,SIZE))
                        l.append(maskedImage)
                        continue
                imgArr=np.array(maskedImage)
                bbox = imgArr_2_objXminYminXmaxYmax(
                    imgArr,
                    bg_color=(255, 255, 255),
                    THRES=5,
                )
                x0,y0,x1,y1=bbox
                assert x0<x1
                assert y0<y1
                if 0:
                    
                    imgArr=draw_bbox(imgArr,bbox,bbox_type='x0y0x1y1',
                                    thickness=max(int(imgArr.shape[0]*0.02),3))
                    # debug_imsave(Path('infer_pairs')/f'{refId}'/path.name,imgArr)
                    maskedImage=imgArr
                else:
                    maskedImage=imgArr[y0:y1,x0:x1,:]
                maskedImage=cv2.resize(maskedImage,(SIZE,SIZE))
                l.append(maskedImage)
            vis_img = cv2_util.concat_images_list(
                *l,img_num_per_row=int(math.sqrt(len(l))),
                vert=0
            )
            debug_imsave(root_config.path_4debug + f"/OMNIDatabase-images_B/{obj}.jpg", vis_img)

        # scy_debug_check_pose_in_database()
        # scy_debug__vis_images_B()

    """
        def __get__id_2_imgName(self,num_img):
            id_2_imgName = {}
            # imgNames = os.listdir(f'{self.__dir}/images')
            imgpaths = glob.glob(f'{self.__dir}/images/*.jpg',)
            imgNames = [os.path.basename(_) for _ in imgpaths]
            ct = 0
            id_ = 0
            while (len(id_2_imgName.keys()) < num_img):
                imgName = f'{ct:05}.jpg'
                if imgName in imgNames:
                    id_2_imgName[id_] = imgName
                    id_ += 1
                ct += 1
                assert ct<9999,'有问题吧'
            print("[OMNIDatabase]id_2_imgName=", id_2_imgName)
            return id_2_imgName

    """
    def __get_poses_Ks(self):
        import numpy as np
        import os
        poses_arr: np.ndarray = np.load(os.path.join(self.__dir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        bds = poses_arr[:, -2:].transpose([1, 0])
        #
        pose34s: np.ndarray = poses[:, :, :4].copy()  
        intrinsicVector31s = poses[:, :, 4].copy()  # [[height, width, focal]*N_img]
        Ks=[]
        for intrinsicVector31 in intrinsicVector31s:
            height, width, focal = intrinsicVector31
            K = np.array([
                [focal, 0, width / 2],
                [0, focal, height / 2],
                [0, 0, 1],
            ])
            Ks.append(K)
        #
        l_pose34: list = [_ for _ in pose34s]
        l_c2w44 = [Pose_R_t_Converter.pose34_2_pose44(pose34) for pose34 in l_pose34]
        l_w2c44 = [np.linalg.inv(c2w44) for c2w44 in l_c2w44]
        # LLFF to OpenCV(colmap)
        Pl2o = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)  # o means OpenCV, p means pytorch3d
        l_w2c44_opencv = [Pl2o @ _ for _ in l_w2c44]
        l_w2c34 = [w2c44[:3, :] for w2c44 in l_w2c44_opencv]
        return l_w2c34, Ks

    def get_image_full_path(self, img_id):
        # return f'{self.__dir}/images/{self.id_2_imgName[img_id]}'
        return f'{self.__dir}/images/{img_id:05}.jpg'

    def get_K(self, img_id):
        return np.copy(self.Ks[self.__img_id_2_originalIndex[img_id]])

    def get_pose(self, img_id):
        return self.poses[self.__img_id_2_originalIndex[img_id]]

    def get_img_ids(self):
        return self._img_ids.copy()

    def get_mask_full_path(self, img_id):
        # return f'{self.__dir}/matting/{self.id_2_imgName[img_id]}'
        return f'{self.__dir}/matting/{img_id:05}.jpg'


class OmniDataset(LinemodDataset):
    def __init__(self, category: str):
        # super().__init__(category)
        self.sequence_list = [""]
        self.database = OMNIDatabase(obj=category)
        print(1)


# if __name__=="__main__":

#     dataset=OmniDataset(category="bed_001")
#     batch=dataset.get_data_4gen6d(sequence_name="",ids=(3,7))
#     print(batch)
def __l_imgNameInt_2_l_id(obj, l_imgNameInt: list):
    def k2v_2_v2k(k2v):
        v2k = {}
        for k, v in k2v.items():
            v2k[v] = k
        return v2k

    try:
        with HiddenPrints():
            database = OMNIDatabase(obj=obj)
    except FileNotFoundError as e:
        print(e)
        return []
    id_2_imgName = database.id_2_imgName
    imgName_2_id = k2v_2_v2k(id_2_imgName)
    l_id = []
    for imgNameInt in l_imgNameInt:
        imgName = f"{imgNameInt:05}.jpg"
        id_ = imgName_2_id[imgName]
        l_id.append(id_)
    print("\n", obj, "\n", str(l_id)[1:-1], "\n", l_id)
    return l_id


