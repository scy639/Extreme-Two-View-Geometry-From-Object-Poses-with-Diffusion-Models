import numpy as np
import os,sys


class Mask:
    @staticmethod
    def hw_255__2__hw_bool(arr:np.ndarray,THRES=125):
        """
        hwAny
        """
        assert arr.dtype==np.uint8
        arr=arr.astype(np.float32)/255
        arr[arr<THRES/255]=0
        arr[arr>=THRES/255]=1
        return arr

    @staticmethod
    def hw0__2__hw1(arr: np.ndarray,):
        assert len(arr.shape)==2
        arr=arr[:,:,None]
        return arr
    @staticmethod
    def hw0__2__hw3(arr: np.ndarray,):
        assert len(arr.shape)==2
        arr=arr[:,:,None]
        arr=np.concatenate([arr,arr,arr],axis=-1)
        assert len(arr.shape)==3 and arr.shape[-1]==3
        return arr
    @staticmethod
    def hw3__2__hw0(arr: np.ndarray,):
        assert len(arr.shape)==3 and arr.shape[-1]==3
        arr_c0=arr[:,:,0]
        arr_c1=arr[:,:,1]
        arr_c2=arr[:,:,2]
        THRES=0
        
        a01 =np.abs(arr_c0-arr_c1)>THRES
        if   np.any(a01):
            print("np.where(a01)",np.where(a01))
            print("corresponding arr_c0", arr_c0[np.where(a01)])
            print("corresponding arr_c1", arr_c1[np.where(a01)])
            assert  0
        a02 =np.abs(arr_c0-arr_c2)>THRES
        if   np.any(a02):
            print("np.where(a02)",np.where(a02))
            print("corresponding arr_c0", arr_c0[np.where(a02)])
            print("corresponding arr_c2", arr_c2[np.where(a02)])
            assert  0
        return arr_c0
    @staticmethod
    def get_blackBg_maskedImage_from_hw0_255( img: np.ndarray,mask: np.ndarray,THRES=125):
        assert len(img.shape)==3
        assert len(mask.shape)==2
        assert img.shape[:2]==mask.shape[:2]
        mask=Mask.hw_255__2__hw_bool(mask,THRES=THRES)
        mask=Mask.hw0__2__hw1(mask)
        assert img.shape[:2]==mask.shape[:2]
        masked_image=img*mask
        return masked_image

    @staticmethod
    def get_blackBg_maskedImage_from_hw1_255(img: np.ndarray, mask: np.ndarray, THRES=125):
        assert len(img.shape) == 3
        assert len(mask.shape) == 2
        mask = Mask.hw_255__2__hw_bool(mask, THRES=THRES)
        masked_image = img * mask
        return masked_image

    @staticmethod
    def get_whiteBg_maskedImage_from_hw0_255( img: np.ndarray,mask: np.ndarray,THRES=125):
        assert len(img.shape)==3
        assert len(mask.shape)==2
        mask=Mask.hw_255__2__hw_bool(mask,THRES=THRES)
        mask=Mask.hw0__2__hw1(mask)
        # mask==1, use img; else use white
        white_image = np.ones_like(img) * 255
        t=img * mask
        tt=np.array(t,dtype=np.uint8)
        t2=white_image * (1 - mask)
        masked_image = img * mask + white_image * (1 - mask)
        return masked_image


    @staticmethod
    def get_whiteBg_maskedImage_from_hw1_255(img: np.ndarray, mask: np.ndarray, THRES=125):
        assert len(img.shape) == 3
        assert len(mask.shape) == 2
        mask = Mask.hw_255__2__hw_bool(mask, THRES=THRES)
        # mask==1, use img; else use white
        white_image = np.ones_like(img) * 255
        masked_image = img * mask + white_image * (1 - mask)
        return masked_image
    @staticmethod
    def rgbaImage__2__hw0_255(img: np.ndarray , ALPHA_THRES=0):
        assert len(img.shape)==3
        assert img.shape[-1]==4
        alpha=img[:,:,3]
        assert np.all(alpha<=255)
        assert np.all(alpha>=0)
        hw0_255=alpha
        hw0_255[hw0_255<=ALPHA_THRES]=0
        hw0_255[hw0_255>ALPHA_THRES]=255
        return hw0_255
    @staticmethod
    def mask_hw0_bool__2__bbox(mask):  # TODO check
        # check all are 0 or 1; check shape
        unique_values = np.unique(mask)
        assert np.array_equal(unique_values, np.array([0, 1]))
        #
        assert len(mask.shape)==2
        #
        mask = np.where(mask == True)
        # x1, y1 = np.min(mask, axis=1)
        # x2, y2 = np.max(mask, axis=1)
        y1, x1 = np.min(mask, axis=1)
        y2, x2 = np.max(mask, axis=1)
        return [x1, y1, x2, y2]