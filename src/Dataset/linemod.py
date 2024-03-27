import mask_util
import root_config
from imports import *
import glob,threading
from pathlib import Path
import cv2
import numpy as np
import os
from PIL import Image
from skimage.io import imread, imsave
import  pickle
import json
import os.path as osp
import random, os
import numpy as np
import torch
import image_util
from PIL import Image, ImageFile
try:
    from .BaseDataset import  BaseDataset
    from .BaseDatabase import  BaseDatabase
except ImportError:
    from BaseDataset import  BaseDataset
    from BaseDatabase import  BaseDatabase
from torchvision import transforms

def read_pickle(pkl_path):#from Gen6d
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pkl_path):#from Gen6d
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)






class LINEMODDatabase(BaseDatabase) :#from Gen6d

    def __init__(self, category):
        self.K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]], dtype=np.float32)
        self.category = category
        # self.img_ids = [str(k) for k in range(len(os.listdir(f'{LINEMOD_ROOT}/{self.category}/JPEGImages')))]
        self.img_ids = [k for k in range(len(os.listdir(f'{LINEMOD_ROOT}/{self.category}/JPEGImages')))]
        self.object_center = np.zeros(3, dtype=np.float32)
        self.object_vert = np.asarray([0, 0, 1], np.float32)
        self.img_id2depth_range = {}
        self.img_id2pose = {}
    def get_image_full_path(self, img_id):
        return f'{LINEMOD_ROOT}/{self.category}/JPEGImages/{int(img_id):06}.jpg'

    def get_K(self, img_id):
        return np.copy(self.K)

    def get_pose(self, img_id):
        if img_id in self.img_id2pose:
            return self.img_id2pose[img_id]
        else:
            pose = np.load(f'{LINEMOD_ROOT}/{self.category}/pose/pose{int(img_id)}.npy')
            self.img_id2pose[img_id] = pose
            return pose

    def get_img_ids(self):
        return self.img_ids.copy()

    def get_mask_full_path(self, img_id):
        return f'{LINEMOD_ROOT}/{self.category}/mask/{int(img_id):04}.png'


class LinemodDataset(BaseDataset):
    def __init__(
            self,
            category:str ,
    ):
        self.sequence_list=[""]
        self.database = LINEMODDatabase(category)
    def __len__(self):
        return len(self.database.get_img_ids())
    def seq_2_l_index(self,seq):
        assert seq == ""
        return self.database.get_img_ids()

    def _get_from_database(
        self, index0, index1,
        q0ipr,q1ipr, max_size=None, mask_out=True,
        image_full_path_TYPE:BaseDataset.ENUM_image_full_path_TYPE=BaseDataset.ENUM_image_full_path_TYPE.raw,
    ):
        """
        只要dataset 有database，就可以
        
        images_not_transformed(,K,bbox,pose):肯定是所有process（比如max_size is None, mask_out false,q0ipr none,那所有process就是空）后的
        image_full_paths:不一定是process后的.depends on:
            1. image_full_path_TYPE
            2. q0ipr. 有q0ipr就一定会based on _Q0...
        """
        database=self.database
        images_not_transformed = []
        image_full_paths = []  # 
        l_bbox = []
        pose44s=[]
        rotations = []
        translation31s = []
        Ks = []
        _index_pair=(index0,index1)
        for j,index in enumerate(_index_pair):
            image_full_path = database.get_image_full_path(index)
            image = Image.open(image_full_path).convert("RGB")
            # save as 'tmp.jpg'
            # image.save('tmp.jpg')
            white_image = Image.new("RGB", image.size, (255, 255, 255))
            # mask = Image.open(database.get_mask_full_path(index)).convert("L")
            mask=database.get_mask_hw0_255(index)
            # if j==0:
            if 1:
                def get_kernel_size(_mask,rela):
                    #
                    _mask = Image.fromarray( _mask > 125)
                    #
                    _bbox = mask_util.Mask.mask_hw0_bool__2__bbox( np.array(_mask))
                    w,h=_bbox[2]-_bbox[0],_bbox[3]-_bbox[1]
                    #
                    kernel_size=min(w,h)
                    kernel_size=rela*kernel_size
                    kernel_size=int(kernel_size)
                    kernel_size=(kernel_size,kernel_size)
                    return kernel_size
                if root_config.MASK_ABLATION=='EROSION':
                    mask=image_util.erode_image(mask,get_kernel_size(mask,0.01),iterations=5)
                elif root_config.MASK_ABLATION=='DILATION':
                    # mask=image_util.dilate_image(mask,get_kernel_size(mask,0.1),iterations=1)
                    # mask=image_util.dilate_image(mask,get_kernel_size(mask,0.05),iterations=1)
                    mask=image_util.dilate_image(mask,get_kernel_size(mask,0.02),iterations=1)
                else:
                    assert root_config.MASK_ABLATION is None
            mask=Image.fromarray(mask)
            if mask.size != image.size:
                mask = mask.resize(image.size)
            mask = Image.fromarray(np.array(mask) > 125)
            if mask_out:
                image = Image.composite(image, white_image, mask)
            # if j==0:
            if 1:
                if root_config.MASK_ABLATION is not None:
                    ttt426=image_full_path
                    image_full_path=f"{image_full_path}_{root_config.MASK_ABLATION[:3]}.png"
                    image.save(image_full_path)
                    print(f"-----MASK_ABLATION------\n  before: {ttt426}\n  after: {image_full_path}\n")
                    del ttt426
            if q0ipr!=0 and j==0:
                mask,_=in_plane_rotate_camera(q0ipr,mask,np.eye(4),fillcolor=0)
            elif   q1ipr!=0 and j==1:
                mask,_=in_plane_rotate_camera(q1ipr,mask,np.eye(4),fillcolor=0)
            bbox = mask_util.Mask.mask_hw0_bool__2__bbox( np.array(mask))
            # image = self._crop_image(image, bbox,
            #                            white_bg=1,
            #                          )
            pose = database.get_pose(index)
            pose = Pose_R_t_Converter.pose34_2_pose44(pose)
            if q0ipr!=0 and j==0:
                image,pose=in_plane_rotate_camera(q0ipr,image,pose)
                image_full_path=f"{image_full_path}_Q0Sipr{q0ipr}.png"
                image.save(image_full_path)
            elif   q1ipr!=0 and j==1:
                image,pose=in_plane_rotate_camera(q1ipr,image,pose)
                image_full_path=f"{image_full_path}_Q1Sipr{q1ipr}.png"
                image.save(image_full_path)
            pose = opencv_2_pytorch3d__leftMulW2cpose(pose)
            rotation = pose[:3, :3]
            translation31 = pose[:3, 3:]
            K=database.get_K(index)
            # resize
            if max_size:
                image=np.array(image)
                #   if size > 320, resize to 320 (dont change aspect ratio
                if image.shape[0]>max_size or image.shape[1]>max_size:
                    h_old=image.shape[0]
                    w_old=image.shape[1]
                    if image.shape[0]>image.shape[1]:
                        h_new=max_size
                        w_new=int(round(max_size/image.shape[0]*image.shape[1]))
                    else:
                        h_new=int(round(max_size/image.shape[1]*image.shape[0] ))
                        w_new=max_size
                    image=cv2.resize(
                        image,(w_new,h_new),
                        interpolation=    cv2.INTER_AREA,
                    )
                    K=CameraMatrixUtil.resize(K=K, h_old=h_old, w_old=w_old, h_new=h_new, w_new=w_new)
                    #bbox 
                    bbox=[int(round(_/h_old*h_new)) for _ in bbox]
                    assert bbox[0]>=0 and bbox[1]>=0 and bbox[2]<=w_new and bbox[3]<=h_new
                    bbox[0]=max(0,bbox[0])
                    bbox[1]=max(0,bbox[1])
                    bbox[2]=min(w_new,bbox[2])
                    bbox[3]=min(h_new,bbox[3])
                    if image_full_path_TYPE==BaseDataset.ENUM_image_full_path_TYPE.resized:
                        #save
                        # image_full_path=Path(image_full_path)
                        # image_full_path=image_full_path.parent.parent/f"max_size={max_size}"/(image_full_path.stem+'.png')
                        image_full_path=image_full_path+f"_MS{max_size}.png"
                        image_full_path=Path(image_full_path)
                        if image_full_path.exists():
                            assert 1,image_full_path
                        image_full_path=str(image_full_path)
                        imsave(image_full_path,image)
                    print(f"after {max_size=}: {image.shape=}, {image_full_path=}, {K=}")
                image=PIL.Image.fromarray(image)
            if    0     :#draw bbox and debug_imsave
                tmp=image.copy()
                draw = PIL.ImageDraw.Draw(tmp)
                draw.rectangle(bbox, outline='red')
                debug_imsave(f"after-{max_size=}/index={index}.png",tmp)
                del tmp
            # append
            images_not_transformed.append(image)
            image_full_paths.append(image_full_path)
            pose44s.append(torch.tensor(pose))
            rotations.append(torch.tensor(rotation))
            translation31s.append(torch.tensor(translation31))
            l_bbox.append(bbox)
            Ks.append(K)
        return images_not_transformed,image_full_paths,rotations,translation31s,pose44s,l_bbox,Ks
    def get_data_4gen6d(self, sequence_name, ids  ,q0ipr=0,q1ipr=0,):
        assert sequence_name == "" or self.__class__.__name__ == 'Co3dv2Dataset'
        assert  len(ids)==2
        """
        only need these field in batch:
            1. image_not_transformed_full_path
            # 2. relative_rotation;relative_t31
            2. relative_pose44
            3. detection_outputs if ... else bbox
            4. K
        """
        images_not_transformed, image_full_paths, rotations, translation31s, pose44s,l_bbox ,Ks= self._get_from_database(ids[0],ids[1],q0ipr,q1ipr,)
        detection_outputs=[]
        def bbox2detection_outputs(bbox,img_w,img_h):
            x1,y1,x2,y2=bbox
            obj_w=x2-x1
            obj_h=y2-y1
            # scale=max(obj_w/img_w,obj_h/img_h)
            scale=max(obj_w/128,obj_h/128)
            ret={
                "positions": [[(bbox[0] + bbox[2]) / 2/img_w, (bbox[1] + bbox[3]) / 2/img_h]],
                "scales": [scale],
            }
            return ret
        for i, (bbox, image) in enumerate(zip(l_bbox, images_not_transformed)):
            w, h = image.width, image.height
            bbox = np.array(bbox)#[x1:int,y1,x2,y2]
            l_bbox[i]=bbox
            detection_outputs.append(bbox2detection_outputs(bbox,img_w=w,img_h=h))
        batch={}
        if(not root_config.LOOK_AT_CROP_OUTSIDE_GEN6D):
            batch["detection_outputs"]=detection_outputs
        else:
            batch["bbox"]=l_bbox
        # batch["R"] = torch.stack(rotations)
        # batch["T"] = torch.stack(translation31s)
        Ks:torch.Tensor = torch.tensor(Ks)
        Ks = Ks.numpy()
        batch["K"] =Ks
        if torch.any(torch.isnan(torch.stack(translation31s))):
            print(ids)
            assert False
        """        
        permutations = ((0,1),(1,0))
        relative_rotation = torch.zeros((2, 3, 3))
        for k, t in enumerate(permutations):
            i, j = t
            relative_rotation[k] =  rotations[j]@rotations[i].T
            # relative_rotation[k] =  rotations[j].T@rotations[i]
        batch["relative_rotation"] = relative_rotation
        """
        relative_pose44 = pose44s[1] @ torch.linalg.inv(pose44s[0])
        batch["relative_pose44"] = relative_pose44
        def vis_4debug():
            tmp_poseVisualizer=PoseVisualizer()
            tmp_poseVisualizer.append_R(R=np.eye(3),color="white")
            tmp_poseVisualizer.append_R(R=rotations[0],color="grey")
            tmp_poseVisualizer.append_R(R=rotations[1],color="g")
            tmp_poseVisualizer.append_R(R=relative_pose44[0][:3,:3],color="yellow")
            # tmp_poseVisualizer.append_R(R=relative_rotation[0].T,color="yellow")
            tmp_img = cv2_util.concat_images_list(
                tmp_poseVisualizer.get_img__mulView_A(
                    vert=0, do_not_show_base=1,
                    y_is_vertical=1,
                    title=f"ids,image_full_paths={ids},\n{[os.path.relpath(_,root_config.path_root) for _ in image_full_paths]}"
                ),
                cv2_util.concat_images_list(
                    *[np.array(img) for img in images_not_transformed],
                    vert=0),
                vert=1,
            )
            # debug_imsave(f'linemod_GT_R-3-y_is_vertical==0/ids={ids}.jpg',tmp_img,)
            debug_imsave(f'omni-get_data_4gen6d/ids={ids}.jpg',tmp_img,)
        # vis_4debug()
        # batch["image_not_transformed"] = images_not_transformed
        def batch_img_2_batch_img_full_path(batch_img):
            batch_img_full_path=[]
            for i in range(2):
                img =batch_img [i]
                img_name_without_dir:str=os.path.basename(image_full_paths[i])
                # img_name_without_suffix = img_name_without_dir.split('.')[0]
                assert img_name_without_dir.endswith('.png') or img_name_without_dir.endswith('.jpg')
                img_name_without_suffix = img_name_without_dir[:-4]
                tmp_cate_or_obj=self.database.cate if self.__class__.__name__=='Co3dv2Dataset' else self.database.obj
                if self.__class__.__name__=='NaviDataset':
                    tmp_cate_or_obj=self.database._obj_with_scene
                """
                
                """
                if self.__class__.__name__=='Co3dv2Dataset':
                    pass
                elif self.__class__.__name__=='NaviDataset':
                    pass
                else:
                    assert img_name_without_dir.endswith('.png'),'database出来的图就是jpg,去看看是不是有损，影响zero123'
                if root_config.SHARE_tmp_batch_images :
                    assert isinstance(root_config.SHARE_tmp_batch_images,str) 
                    tmp_batch_images_subfolderName=root_config.SHARE_tmp_batch_images
                else:
                    assert root_config.SHARE_tmp_batch_images is False
                    tmp_batch_images_subfolderName=f'{os.getpid()}-{threading.currentThread().getName()}'
                full_path = os.path.join(root_config.path_root, f'evaluate/tmp_batch_images/{tmp_batch_images_subfolderName}/[{root_config.DATASET}][{tmp_cate_or_obj}][{sequence_name}]{img_name_without_suffix}{root_config.tmp_batch_image__SUFFIX}')
                os.makedirs(os.path.dirname(full_path),exist_ok=True)
                pDEBUG("get_data path:", full_path)
                if root_config.SHARE_tmp_batch_images and os.path.exists(full_path):
                    pDEBUG(f"root_config.SHARE_tmp_batch_images and os.path.exists(full_path) => do not save again")
                else:
                    img.save(full_path)
                batch_img_full_path.append(full_path)
            return batch_img_full_path
        batch["image_not_transformed_full_path"]=batch_img_2_batch_img_full_path(images_not_transformed)
        pDEBUG("batch[image_not_transformed_full_path]:", batch["image_not_transformed_full_path"])
        return batch