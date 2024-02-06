import numpy as np
from PIL import Image
import cv2

def show_img(img):
    if isinstance(img, np.ndarray):
        
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif isinstance(img, Image.Image):
        
        img.show()
    elif isinstance(img, str):
        
        image = cv2.imread(img)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Unsupported image type.")

def save_img(img, path):
    if isinstance(img, np.ndarray):
        
        cv2.imwrite(path, img)
    elif isinstance(img, Image.Image):
        
        img.save(path)
    else:
        print("Unsupported image type.")