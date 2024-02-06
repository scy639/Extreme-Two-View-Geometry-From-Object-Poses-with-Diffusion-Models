import cv2
import numpy as np
from typing import Tuple, Optional
from skimage.io import imread, imsave

def add_text_to_image(
        image_rgb: np.ndarray,
        label: str,
        top_left_xy: Tuple = (0, 0),
        font_scale: float = 1,
        font_thickness: float = 1,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        font_color_rgb: Tuple = (0, 0, 255),
        bg_color_rgb: Optional[Tuple] = None,
        outline_color_rgb: Optional[Tuple] = None,
        line_spacing: float = 1,
):
    """
    from https://stackoverflow.com/questions/27647424/opencv-puttext-new-line-character
    """
    """
    Adds text (including multi line text) to images.
    You can also control background color, outline color, and line spacing.

    outline color and line spacing adopted from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    """
    OUTLINE_FONT_THICKNESS = 3 * font_thickness

    im_h, im_w = image_rgb.shape[:2]

    for line in label.splitlines():
        x, y = top_left_xy

        # ====== get text size
        if outline_color_rgb is None:
            get_text_size_font_thickness = font_thickness
        else:
            get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

        (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            font_face,
            font_scale,
            get_text_size_font_thickness,
        )
        line_height = line_height_no_baseline + baseline

        if bg_color_rgb is not None and line:
            # === get actual mask sizes with regard to image crop
            if im_h - (y + line_height) <= 0:
                sz_h = max(im_h - y, 0)
            else:
                sz_h = line_height

            if im_w - (x + line_width) <= 0:
                sz_w = max(im_w - x, 0)
            else:
                sz_w = line_width

            # ==== add mask to image
            if sz_h > 0 and sz_w > 0:
                bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                bg_mask[:, :] = np.array(bg_color_rgb)
                image_rgb[
                y: y + sz_h,
                x: x + sz_w,
                ] = bg_mask

        # === add outline text to image
        if outline_color_rgb is not None:
            image_rgb = cv2.putText(
                image_rgb,
                line,
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                outline_color_rgb,
                OUTLINE_FONT_THICKNESS,
                cv2.LINE_AA,
            )
        # === add text to image
        image_rgb = cv2.putText(
            image_rgb,
            line,
            (x, y + line_height_no_baseline),  # putText start bottom-left
            font_face,
            font_scale,
            font_color_rgb,
            font_thickness,
            cv2.LINE_AA,
        )
        top_left_xy = (x, y + int(line_height * line_spacing))

    return image_rgb

def putText (img, text, org, fontFace, fontScale, color, thickness=1, lineType=None, bottomLeftOrigin=None):
    line_spacing=1
    top_left_xy=org
    """
    func that wrap cv2.putText (arg and ret keep the same), but support auto line break
    """
    OUTLINE_FONT_THICKNESS = 3 * thickness
    im_h, im_w = img.shape[:2]
    for line in text.splitlines():
        x, y = top_left_xy
        get_text_size_font_thickness = OUTLINE_FONT_THICKNESS
        (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            fontFace,
            fontScale,
            get_text_size_font_thickness,
        )
        line_height = line_height_no_baseline + baseline
        # === add text to image
        img = cv2.putText(
            img,
            line,
            (x, y + line_height_no_baseline),  # putText start bottom-left
            fontFace,
            fontScale,
            color,
            thickness=thickness,
            # lineType=cv2.LINE_AA,
            lineType=lineType,
            bottomLeftOrigin=bottomLeftOrigin,
        )
        top_left_xy = (x, y + int(line_height * line_spacing))
    return img
def putText_B(img, text, org=(5,5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 100, 100), thickness=1, lineType=None, bottomLeftOrigin=None):
    """
    provide many default args
    """
    return putText(
        img,
        text,
        org, fontFace, fontScale, color, thickness,
        lineType, bottomLeftOrigin,
    )
"""
from gen6d
"""
def concat_images(img0, img1, vert=False):
    if not vert:
        h0, h1 = img0.shape[0], img1.shape[0],
        if h0 < h1: img0 = cv2.copyMakeBorder(img0, 0, h1 - h0, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)
        if h1 < h0: img1 = cv2.copyMakeBorder(img1, 0, h0 - h1, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)
        img = np.concatenate([img0, img1], axis=1)
    else:
        w0, w1 = img0.shape[1], img1.shape[1]
        if w0 < w1: img0 = cv2.copyMakeBorder(img0, 0, 0, 0, w1 - w0, borderType=cv2.BORDER_CONSTANT, value=0)
        if w1 < w0: img1 = cv2.copyMakeBorder(img1, 0, 0, 0, w0 - w1, borderType=cv2.BORDER_CONSTANT, value=0)
        img = np.concatenate([img0, img1], axis=0)

    return img


def concat_images_list(*args, vert=False,max_h=None,max_w=None,img_num_per_row=None):
    if len(args) == 1: return args[0]
    if img_num_per_row:
        if not len(args)%img_num_per_row==0:
            args=args+tuple([np.array([[[0,0,0]]])]*(img_num_per_row-len(args)%img_num_per_row))
        args=[concat_images_list(*args[i:i+img_num_per_row],vert=vert,max_h=max_h,max_w=max_w) for i in range(0,len(args),img_num_per_row)]
        return concat_images_list(*args,vert=not vert,max_h=max_h,max_w=max_w)
    if(max_h is not None):
        args=[cv2.resize(img,(int(img.shape[1]*max_h/img.shape[0]),max_h))  if img.shape[0]>max_h else img for img in args]
    if(max_w is not None):
        args=[cv2.resize(img,(max_w,int(img.shape[0]*max_w/img.shape[1])))  if img.shape[1]>max_w else img for img in args]
    img_out = args[0]
    for img in args[1:]:
        img_out = concat_images(img_out, img, vert)
    return img_out
if(__name__=="__main__"):
    img=np.zeros((100,100,3),np.uint8)
    img=putText(img,"hello\nworld",(img.shape[1]-50,0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
    cv2.imshow("img",img)
    cv2.waitKey(0)
if(__name__=="__main__"):
    img=np.zeros((500,300,3),np.uint8)
    img=putText(img,"hello\nworld",(0,0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
    cv2.imshow("img",img)
    cv2.waitKey(0)