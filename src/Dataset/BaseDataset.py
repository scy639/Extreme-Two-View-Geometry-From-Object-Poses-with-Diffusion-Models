import gzip
import json
import os.path as osp
import random, os

import numpy as np
import torch
from PIL import Image, ImageFile
from pytorch3d.renderer import PerspectiveCameras
# from torch.utils.data import Dataset
from torchvision import transforms
# try:
#     from utils.bbox import square_bbox
#     # from utils.misc import get_permutations
#     from utils.normalize_cameras import first_camera_transform, normalize_cameras
# except ModuleNotFoundError:
#     from ..utils.bbox import square_bbox
#     # from ..utils.misc import get_permutations
#     from ..utils.normalize_cameras import first_camera_transform, normalize_cameras




class BaseDataset:
    class ENUM_image_full_path_TYPE:
        raw=0
        resized=1
    @staticmethod
    def _crop_image( image, bbox, white_bg=False):
        if white_bg:
            # Only support PIL Images
            image_crop = Image.new(
                "RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255)
            )
            image_crop.paste(image, (-bbox[0], -bbox[1]))
        else:
            image_crop = transforms.functional.crop(
                image,
                top=bbox[1],
                left=bbox[0],
                height=bbox[3] - bbox[1],
                width=bbox[2] - bbox[0],
            )
        return image_crop
    def __init__(self):
        self.sequence_list=[]
    def __len__(self):
        return len(self.sequence_list)
    def _jitter_bbox(self, bbox):
        from utils.bbox import square_bbox
        bbox = square_bbox(bbox.astype(np.float32))

        s = np.random.uniform(self.jitter_scale[0], self.jitter_scale[1])
        tx, ty = np.random.uniform(self.jitter_trans[0], self.jitter_trans[1], size=2)

        side_length = bbox[2] - bbox[0]
        center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side_length
        extent = side_length / 2 * s

        # Final coordinates need to be integer for cropping.
        ul = (center - extent).round().astype(int)
        lr = ul + np.round(2 * extent).astype(int)
        return np.concatenate((ul, lr))
    def _get_batch_4_relposepp(
            self,
            transform,
            #
            images,
            image_full_paths,
            # rotations,
            # translation31s,
            pose44s,
            l_bbox,
    ):
        """
        需要这些字段：
        batch:{
            images_transformed
            # relative_rotation;relative_t31
            relative_pose44
            crop_params
        }
        """
        self.jitter_scale = [1.15, 1.15]
        self.jitter_trans = [0, 0]
        crop_parameters = []
        images_transformed = []
        l_bbox=[np.array(bbox) for bbox in l_bbox]
        for i, ( image, bbox) in enumerate(zip( images, l_bbox)):
            if transform is None:  # not None
                images_transformed.append(image)
            else:
                w, h = image.width, image.height
                bbox_jitter = self._jitter_bbox(bbox)

                image = self._crop_image(image, bbox_jitter,
                                        #  white_bg=self.mask_images
                                           white_bg=1
                                         )
                images_transformed.append(transform(image))

                crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
                cc = (2 * crop_center / min(h, w)) - 1
                crop_width = 2 * (bbox_jitter[2] - bbox_jitter[0]) / min(h, w)

                crop_parameters.append(
                    torch.tensor([-cc[0], -cc[1], crop_width]).float()
                )

        batch = {
            # "len": 2,
            # "ind": torch.tensor([index0,index1]),
            "crop_params": torch.stack(crop_parameters)
        }
        self.normalize_cameras=0# TODO
        if self.normalize_cameras:  # True
            cameras = PerspectiveCameras(
                focal_length=[annoDic["focal_length"] for annoDic in annoDic_01],
                principal_point=[annoDic["principal_point"] for annoDic in annoDic_01],
                R=[annoDic["R"] for annoDic in annoDic_01],
                T=[annoDic["T"] for annoDic in annoDic_01],
            )

            normalized_cameras, _, _, _, _ = normalize_cameras(cameras)

            if self.first_camera_transform or self.first_camera_rotation_only:
                normalized_cameras = first_camera_transform(
                    normalized_cameras,
                    rotation_only=self.first_camera_rotation_only,
                )

            if normalized_cameras == -1:
                print("Error in normalizing cameras: camera scale was 0")
                assert False

            batch["R"] = normalized_cameras.R
            batch["T"] = normalized_cameras.T
            # batch["R_original"] = torch.stack(
            #     [torch.tensor(annoDic["R"]) for annoDic in annoDic_01]
            # )
            # batch["T_original"] = torch.stack(
            #     [torch.tensor(annoDic["T"]) for annoDic in annoDic_01]
            # )

            if torch.any(torch.isnan(batch["T"])):
                # print(ids)
                print(category)
                print(sequence_name)
                assert False

        else:
            # batch["R"] = torch.stack(rotations)
            # batch["T"] = torch.stack(translation31s)
            pass

        """
        permutations = ((1, 0), (0, 1))
        relative_rotation = torch.zeros((2, 3, 3))
        relative_t31 = torch.zeros((2, 3, 1))
        for k, t in enumerate(permutations):
            i, j = t
            relative_rotation[k] = rotations[i].T @ rotations[j]
            relative_t31[k] = translation31s[i].T @ translation31s[j]
        batch["relative_rotation"] = relative_rotation
        batch["relative_t31"] = relative_t31
        """
        relative_pose44 = pose44s[1] @ torch.linalg.inv(pose44s[0])
        batch["relative_pose44"] = relative_pose44
        # Add images
        if transform is None:
            batch["images_transformed"] = images
        else:
            batch["images_transformed"] = torch.stack(images_transformed)
        return batch
    def get_data(self, sequence_name, index0,index1,   ):
        """
        需要这些字段：
        batch:{
            images_transformed
            relative_rotation;relative_t31
            crop_params
        }
        """
        pass

    def get_data_4gen6d(self, index=None, sequence_name=None, ids=(0, 1), no_images=False):  
        """
        only need these field in batch:
            1. image_not_transformed_full_path
            2. relative_rotation;relative_t31
            3. detection_outputs if ... else bbox
            4. K
        """
        pass
