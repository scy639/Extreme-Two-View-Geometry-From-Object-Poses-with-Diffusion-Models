def main(read_path, save_path):
    bg_color=(255,255,255)

    from image_util  import  imgArr_2_objXminYminXmaxYmax
    import os
    import cv2
    """
    read_path下所有.jpg，识别出obj XminYminXmaxYmax后，crop出obj，然后在obj周围补bg_color使size不变，保存至save_path/jpg_name
    """
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
    for jpg_name in os.listdir(read_path):
        if jpg_name.endswith(".jpg"):
            #print(jpg_name)
            img = cv2.imread(os.path.join(read_path, jpg_name))
            h,w=img.shape[:2]
            xmin, ymin, xmax, ymax = imgArr_2_objXminYminXmaxYmax(img,bg_color)
            obj = img[ymin:ymax, xmin:xmax]
            obj = cv2.copyMakeBorder(obj, (h-(ymax-ymin))//2,  (h-(ymax-ymin))//2, (w-(xmax-xmin))//2, (w-(xmax-xmin))//2, cv2.BORDER_CONSTANT, value=bg_color)
            obj = cv2.copyMakeBorder(obj, 0, h-obj.shape[0],0,w-obj.shape[1], cv2.BORDER_CONSTANT, value=bg_color)
            cv2.imwrite(os.path.join(save_path, jpg_name), obj)
            

    
    
