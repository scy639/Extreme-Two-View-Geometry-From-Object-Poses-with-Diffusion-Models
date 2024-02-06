import numpy as np
import os,sys,cv2
import PIL
from PIL import Image

def to__image_in_npArr(img):
    """
    convert PIL/np.ndarray type image to np.ndarray
    """
    if isinstance(img, np.ndarray):
        return img
    elif isinstance(img, PIL.Image.Image):
        return np.array(img)
    else:
        raise TypeError("img should be PIL.Image or np.ndarray, got {}".format(type(img)))
def imgArr_2_objXminYminXmaxYmax(imgArr, bg_color,THRES=5):
    """
    param:
        imgArr: np.array
        bg_color: 背景颜色，形如 (R, G, B) 的元组
    return:
        xmin,ymin,xmax,ymax (type= primitive int,NOT np int)
    """
    img_array = imgArr

    
    diff_pixels = np.any(np.abs(img_array - np.array(bg_color)) > THRES, axis=2)

    
    rows = np.any(diff_pixels, axis=1)
    cols = np.any(diff_pixels, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    xmin=xmin.item()
    ymin=ymin.item()
    xmax=xmax.item()
    ymax=ymax.item()
    return xmin, ymin, xmax, ymax
def draw_bbox(img, bbox, color=None, thickness=2,bbox_type='x0y0wh'):
    """
    xmin,ymin,xmax,ymax
    """
    img = np.copy(img)
    if color is not None:
        color = [int(c) for c in color]
    else:
        color = (0, 255, 0)
    if bbox_type=='x0y0wh':
        left = int(round(bbox[0]))
        top = int(round(bbox[1]))
        width = int(round(bbox[2]))
        height = int(round(bbox[3]))
    elif bbox_type=='x0y0x1y1':
        left,top,right,bottom=bbox
        width = right-left
        height = bottom-top
    img = cv2.rectangle(img, (left, top), (left + width, top + height), color, thickness=thickness)
    return img




def print_image_statistics(image):
    """
    Print image statistics:
        type
        dtype and shape
        min, max, mean, median, unique values for each channel
    """
    string = "----[statistics]----\n"
    string += f"type = {type(image)}\n"
    image = to__image_in_npArr(image)
    string += f"dtype = {image.dtype}\n"
    string += f"shape = {image.shape}\n"

    if len(image.shape) == 2:
        channels = [image]
    else:
        # channels = np.split(image, image.shape[-1], axis=-1)#poe generated, I cannot understand easily
        channels = [image[:, :, i] for i in range(image.shape[-1])]

    for i, channel in enumerate(channels):
        string += f"\nChannel {i }:\n"
        string += f"  Min: {np.min(channel)}\n"
        string += f"  Max: {np.max(channel)}\n"
        string += f"  Mean: {np.mean(channel)}\n"
        string += f"  Median: {np.median(channel)}\n"
        uniques=np.unique(channel)
        _N=6
        if len(uniques)>_N:
            s_uniques=f"{uniques[:_N//2]}".replace(']','')
            s_uniques+=',...,'
            s_uniques+=f'{uniques[-_N//2:]}'.replace('[','')
        string += f"  Unique values: {s_uniques}\n"
    string=string.replace('\n','\n|')
    string += "----[statistics]over----\n"
    print(string)

def pad_around_center(img, new_size,  ):
    """
    Pad image to a new size with fill color around image center.
    pad with white (255)
    """
    img = to__image_in_npArr(img)
    assert len(img.shape) == 3
    assert len(new_size) == 2

    # compute padding
    height, width, _ = img.shape
    new_height, new_width = new_size
    assert new_height >= height
    assert new_width >= width
    pad_height = new_height - height
    pad_width = new_width - width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # pad image
    img = np.pad(
        img,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=255,
    )
    return img

def rotate_B(degree_clockwise, pilImage: Image.Image,
             only_get_obj_hw=False,
             d_hw:tuple=None,#if not None, then pad img to d_hw
             ):
    """
    逆时针旋转图片 degree_clockwise °
    version B: (没有A，可以将pilImage.rotate就理解为A) 
        1. 假定物体背景为白色
        2. 先pad image (d size= 原图对角线长度),fill color=white;then rotate, detect obj bbox,  crop obj out. if d_hw not None, then pad img to d_hw
    """
    assert -360 <= degree_clockwise <= 360
    assert isinstance(pilImage,Image.Image)
    # img_rot: Image.Image = pilImage.rotate(
    
    #     fillcolor=fillcolor,
    #     resample=Image.BICUBIC,
    # )
    dsize=int(np.linalg.norm(pilImage.size))
    # step1. pad
    # img_padded = Image.new('RGB', (dsize, dsize), color=(255, 255, 255))
    # img_padded.paste(pilImage, (round((dsize-pilImage.size[0])/2), round((dsize-pilImage.size[1])/2)))
    img_padded=pad_around_center(pilImage,(dsize,dsize))
    img_padded=Image.fromarray(img_padded)
    # step2. rotate
    img_rot = img_padded.rotate(degree_clockwise, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    
    xmin, ymin, xmax, ymax = imgArr_2_objXminYminXmaxYmax(to__image_in_npArr(img_rot), bg_color=(255, 255, 255))
    if only_get_obj_hw:
        return ymax-ymin, xmax-xmin
    else:
        img_rot = img_rot.crop((xmin, ymin, xmax, ymax))
        if d_hw is None:
            return img_rot
        else:
            #pad to original size
            # img_padded = Image.new('RGB', pilImage.size, color=(255, 255, 255))
            # img_padded.paste(img_rot, (round((pilImage.size[0]-img_rot.size[0])/2), round((pilImage.size[1]-img_rot.size[1])/2)))
            img_padded=pad_around_center(img_rot,d_hw)
            img_padded=Image.fromarray(img_padded)
            return img_padded

    
def get__max_bbox_size__after_rotates(img,l_rotates_angle:list):
    """
    param:
        img: PIL.Image.Image
        l_rotates_angle:  逆时针旋转角度 deg
    return:
        max_bbox_size: int
    """
    assert isinstance(img,Image.Image)
    assert isinstance(l_rotates_angle,list)
    assert len(l_rotates_angle)>0
    max_bbox_size=0
    for angle in l_rotates_angle:
        h,w=rotate_B(angle,img,only_get_obj_hw=True)
        max_bbox_size=max(max_bbox_size,h,w)
    return max_bbox_size


def rotate_C(
    pilImage: Image.Image,
    l_degree_clockwise: list,  
)->list[Image.Image]:
    
    assert isinstance(pilImage,Image.Image)
    max_bbox_size=get__max_bbox_size__after_rotates(pilImage,l_degree_clockwise)
    l_img_rot=[]
    for degree_clockwise in l_degree_clockwise:
        img_rot=rotate_B(degree_clockwise,pilImage,d_hw=(max_bbox_size,max_bbox_size))
        l_img_rot.append(img_rot)
    return l_img_rot




def erode_image(image, kernel_size,iterations=1):
    """
    Applies erosion to an image using a given kernel size.

    Args:
        image (numpy.ndarray): The input image.
        kernel_size (tuple): The size of the erosion kernel in the format (height, width).

    Returns:
        numpy.ndarray: The eroded image.
    """
    erosion_kernel = np.ones(kernel_size, np.uint8)
    eroded_image = cv2.erode(image, erosion_kernel, iterations=iterations)
    return eroded_image

def dilate_image(image, kernel_size,iterations=1):
    """
    Applies dilation to an image using a given kernel size.

    Args:
        image (numpy.ndarray): The input image.
        kernel_size (tuple): The size of the dilation kernel in the format (height, width).

    Returns:
        numpy.ndarray: The dilated image.
    """
    dilation_kernel = np.ones(kernel_size, np.uint8)
    dilated_image = cv2.dilate(image, dilation_kernel, iterations=iterations)
    return dilated_image