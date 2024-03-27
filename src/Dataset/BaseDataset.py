import json
import os.path as osp
import random, os

import numpy as np
import torch
from PIL import Image, ImageFile
# from torch.utils.data import Dataset
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

    def __init__(self):
        self.sequence_list=[]
    def __len__(self):
        return len(self.sequence_list)


    def get_data_4gen6d(self, index=None, sequence_name=None, ids=(0, 1), no_images=False):  
        """
        only need these field in batch:
            1. image_not_transformed_full_path
            2. relative_rotation;relative_t31
            3. detection_outputs if ... else bbox
            4. K
        """
        pass
