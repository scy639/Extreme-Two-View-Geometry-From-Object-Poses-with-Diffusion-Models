import  root_config
import os
import fcntl
import cv2
from skimage.io import imsave
from import_util import is_in_sysPath,can_not_relative_import
if(can_not_relative_import(__file__)):
    from util_4_e2vg import ImagePathUtil
    from util_4_e2vg import CameraMatrixUtil
    from util_4_e2vg.IntermediateResult import IntermediateResult
else:
    from . import ImagePathUtil
    from . import CameraMatrixUtil
    from  .IntermediateResult import IntermediateResult

# import ImagePathUtil
# import CameraMatrixUtil
# from IntermediateResult import IntermediateResult



import numpy as np
from PIL import Image
from image_util import imgArr_2_objXminYminXmaxYmax




def crop_object(img, bg_color, h, w, K, keep_ratio=True, **kw):
    x = None
    y = None
    x_end = None
    y_end = None
    """
    param:
        img: PIL Image 对象
        bg_color: 背景颜色，形如 (R, G, B) 的元组
        size: 裁剪后的目标尺寸，形如 (width, height) 的元组
    return:
        cropped_img: 裁剪后的物体图像，PIL Image 对象
    """
    
    img_array = np.array(img)
    xmin, ymin, xmax, ymax = imgArr_2_objXminYminXmaxYmax(img_array, bg_color)
    
    obj_width = xmax - xmin
    obj_height = ymax - ymin
    # margin
    margin_w_px = obj_width * kw["margin_percent"]
    margin_h_px = obj_height * kw["margin_percent"]
    obj_width = obj_width + margin_w_px * 2
    obj_height = obj_height + margin_h_px * 2

    if (not kw["norm_obj_by_z"]):
        z4normObj = None
        fx, fy = CameraMatrixUtil.get_fx_fy_4_normObj(z=1,
                                                      obj_w_pixel=obj_width, obj_h_pixel=obj_height,
                                                      img_w=img.width, img_h=img.height)
        K[0][0], K[1][1] = fx, fy
    # ---------debug
    #print(f"cropped_img_array.shape={img_array.shape}")
    #print("xmin,xmax,ymin,ymax=", xmin, xmax, ymin, ymax)
    tmp_img_array = img_array.copy()
    # [cv2.circle(tmp_img_array, tuple(pt), radius=4, color=(255, 0, 0), thickness=-1) for pt in
    cv2.circle(tmp_img_array, tuple((xmin, ymin)), radius=2, color=(255, 0, 0), thickness=-1)
    cv2.circle(tmp_img_array, tuple((xmax, ymax)), radius=2, color=(255, 255, 0), thickness=-1)
    """
    tmp_path="4debug/ttt437"
    if(not os.path.exists(tmp_path)):
        os.makedirs(tmp_path)
    imsave(f"{tmp_path}/rawName={kw['file_name']}", tmp_img_array)
    """

    if (kw["do_not_crop"]):#2023.10: True
        cropped_img = img
    else:
        if (keep_ratio):
            
            target_width = w
            target_height = h
            if (obj_width / obj_height > target_width / target_height):
                adjusted_width = obj_width
                adjusted_height = adjusted_width * target_height / target_width
            else:
                adjusted_height = obj_height
                adjusted_width = adjusted_height * target_width / target_height

            #
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            
            x = x_center - adjusted_width / 2
            y = y_center - adjusted_height / 2
            
            x_end = x_center + adjusted_width / 2
            y_end = y_center + adjusted_height / 2

            
            x = max(0, x)
            x_end = min(img_array.shape[1], x_end)
            adjusted_width = x_end - x
            if (keep_ratio):  
                adjusted_height = adjusted_width * target_height / target_width
                y = y_center - adjusted_height / 2
                y_end = y_center + adjusted_height / 2
            y = max(0, y)
            y_end = min(img_array.shape[0], y_end)
            adjusted_height = y_end - y
            if (keep_ratio):
                adjusted_width = adjusted_height * target_width / target_height
                x = x_center - adjusted_width / 2
                x_end = x_center + adjusted_width / 2

            # to int
            x = max(int(x), 0)
            y = max(int(y), 0)
            x_end = min(int(x_end), img_array.shape[1])
            y_end = min(int(y_end), img_array.shape[0])
            adjusted_width = x_end - x
            adjusted_height = y_end - y
            cropped_img_array = img_array[y:y_end, x:x_end, :]
            xmin = x
            ymin = y
            xmax = x_end
            ymax = y_end
        else:
            
            cropped_img_array = img_array[ymin:ymax, xmin:xmax, :]
        
        cropped_img = Image.fromarray(cropped_img_array)
        if (kw["DRAW_cropped_img"]):
            #print(kw["file_name"])
            #print(f"cropped_img_array.shape={cropped_img_array.shape}")
            #print("x,y,x_end,y_end=", x, y, x_end, y_end)
            # tmp_img_array = cropped_img_array.copy()
            tmp_img_array = img_array.copy()
            # [cv2.circle(tmp_img_array, tuple(pt), radius=4, color=(255, 0, 0), thickness=-1) for pt in
            cv2.circle(tmp_img_array, tuple((xmin, ymin)), radius=4, color=(255, 0, 0), thickness=-1)
            cv2.circle(tmp_img_array, tuple((xmax, ymax)), radius=4, color=(255, 255, 0), thickness=-1)
            imsave(f"4debug/cropped_img/rawName={kw['file_name']}", tmp_img_array)
        K = CameraMatrixUtil.crop(K=K, coord0=(xmin, ymin), coord1=(xmax, ymax))
        
        cropped_img = cropped_img.resize((w, h))
        K = CameraMatrixUtil.resize(K=K, h_old=ymax - ymin, w_old=xmax - xmin, h_new=h, w_new=w)
    if (kw["norm_obj_by_z"]):
        z4normObj = CameraMatrixUtil.get_z_4_normObj(fx=K[0][0], fy=K[1][1],
                                                     obj_w_pixel=obj_width, obj_h_pixel=obj_height,
                                                     img_w=img.width, img_h=img.height)

    return cropped_img, K, z4normObj


import os
from PIL import Image
def calibrate(x,y,calib_xy):
    """
    zero123里x,y是相对你给的图的x,y;而你输入的图不一定是正的.calib_xy为使得输出图为你希望的'pose原点'的输入x,y
    """
    return x-calib_xy[0],y-calib_xy[1]
def crop(read_path, save_path,calib_xy:tuple,base_xyz:tuple,K, **kw):
    assert K=="弃用。从pipeline __run_zero123 一路传进来的这个K应该是zero123生成图的K（crop修改后，最终被用作gen6d里ref database的K(被写进intermediateResult.json然后在gen6d ref database里被读取)）"
    """
    裁剪 read_path 下所有 JPG 图像，并保存至 save_path

    参数：
    - read_path: 图像文件夹的路径
    - save_path: 保存裁剪后图像的文件夹路径
    """
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:  
        if (kw["ask"] and input(f"save_path {save_path} exists. 输入空字符(直接enter)跳过本步骤(crop)") == ""):
            return

    
    image_files = [f for f in os.listdir(read_path) if f.endswith('.jpg')]
    # image_files=image_files[30:40]
    intermediateResult = IntermediateResult()
    
    for image_file in image_files:
        i, j, x, y, z = ImagePathUtil.parse_path(path=image_file)
        #calib
        x, y=calibrate(x, y, calib_xy)
        #
        x,y,z=x+base_xyz[0],y+base_xyz[1],z+base_xyz[2]
        
        image_path = os.path.join(read_path, image_file)
        img = Image.open(image_path)

        
        bg_color = (255, 255, 255)  
        
        K = CameraMatrixUtil.get_K(img_h=img.height, img_w=img.width)
        cropped_img, K, z4normObj = crop_object(img, bg_color, img.height, img.width, K, True, **kw,
                                                file_name=image_file,
                                                )

        if (kw["norm_obj_by_z"]):
            pose = CameraMatrixUtil.xyz2pose(x, y, z4normObj)
        else:
            pose = CameraMatrixUtil.xyz2pose(x, y, z)
        i2=i
        if(root_config.USE_ALL_SAMPLE):
            i2=j*root_config.SAMPLE_NUM+i
        
        save_file = os.path.join(save_path, f"{i2}.jpg")
        if (kw["save_image"]):
            # cropped_img.save(save_file)
            with open(save_file, 'w') as file:#thread safe
                
                fcntl.flock(file, fcntl.LOCK_EX)
                
                cropped_img.save(file)
                
                fcntl.flock(file, fcntl.LOCK_UN)
            #print("save_file full path:", os.path.abspath(save_file))
        intermediateResult.append(i=i2, K=K, pose=pose)
    return intermediateResult
