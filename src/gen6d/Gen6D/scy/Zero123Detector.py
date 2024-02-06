from   .DebugUtil  import   *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
import os
from PIL import Image


class Zero123Detector:
    def __init__(s):
        s.ref_h = None
        s.ref_w = None

    def load_ref_imgs(s, ref_imgs):
        """
        @param ref_imgs: [an,rfn,h,w,3] in numpy
        @return:
        """
        ref_imgs = torch.from_numpy(ref_imgs).permute(0, 3, 1, 2)  # rfn,3,h,w
        rfn, _, h, w = ref_imgs.shape
        s.ref_h = h
        s.ref_w = w

    def detect_que_imgs(s, que_imgs):
        imgs = que_imgs
        """
        io is same as raw gen6d detector:
        @param que_imgs: [qn,h,w,3]
        @return:
        """
        positions = []
        scales = []

        def crop(img, bg_color, h, w,  **kw):
            """
            param:
                img: PIL Image 对象
                bg_color: 背景颜色，形如 (R, G, B) 的元组
                size: 裁剪后的目标尺寸，形如 (width, height) 的元组
            return:
                cropped_img: 裁剪后的物体图像，PIL Image 对象
            """
            
            img_array = np.array(img)

            
            diff_pixels = np.any(np.abs(img_array - np.array(bg_color)) > 5, axis=2)

            
            rows = np.any(diff_pixels, axis=1)
            cols = np.any(diff_pixels, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            obj_width = xmax - xmin
            obj_height = ymax - ymin
            # margin
            margin_w_px = obj_width * kw["margin_percent"]
            margin_h_px = obj_height * kw["margin_percent"]
            obj_width = obj_width + margin_w_px * 2
            obj_height = obj_height + margin_h_px * 2
            
            target_width = w
            target_height = h
            if (obj_width / obj_height > target_width / target_height):
                adjusted_width = obj_width
                adjusted_height = obj_width * target_height / target_width
            else:
                adjusted_width = obj_height * target_width / target_height
                adjusted_height = obj_height

            
            x = xmin + (obj_width - adjusted_width) / 2
            y = ymin - (adjusted_height - obj_height) / 2  

            
            x_end = x + adjusted_width
            y_end = y + adjusted_height

            
            x = max(0, x)
            y = max(0, y)
            x_end = min(img_array.shape[1], x_end)
            y_end = min(img_array.shape[0], y_end)
            # to int
            x = int(x)
            y = int(y)
            x_end = int(x_end)
            y_end = int(y_end)
            def get_crop(img_array, x, y, x_end, y_end):
                cropped_img_array = img_array[y:y_end, x:x_end, :]
                
                cropped_img = Image.fromarray(cropped_img_array)
                
                cropped_img = cropped_img.resize((w, h))
                return cropped_img
            CROP = False
            if (CROP):
                return  get_crop(img_array, x, y, x_end, y_end)
            else:
                position = np.asarray([(x+x_end)/2,(y+y_end)/2])
                scale = adjusted_width/target_width
                return  position,scale
            show_img(get_crop(img_array, x, y, x_end, y_end))

        for img in imgs:
            position,scale=crop(img, (255,255,255), s.ref_h, s.ref_w, margin_percent=0.1)
            positions.append(position)
            scales.append(scale)
        detection_outputs = {
            "positions": positions,
            "scales": scales,
        }
        return detection_outputs
