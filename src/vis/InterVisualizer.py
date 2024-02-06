from .cv2_util import  *
import   cv2


class InterVisualizer:
    def __init__(self, ):
        self.imgs = []
        self.l_text = []

    def append(self, img=None, text=None, row=None, column=None, kw_putText={}):
        assert column==None
        if (img):
            if (text):
                kw_putText = {
                    "org": (10, 30),
                    "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
                    "fontScale": 0.6,
                    "color": (0, 0, 0),
                    # thickness,
                    # lineType,
                    **kw_putText,
                }
                putText(
                    img,
                    text,
                    **kw_putText,
                )
            if(not row):
                row=len(self.imgs)
            while(len(self.imgs)-1<row):
                self.imgs.append([])
            self.imgs[row].append(img)
        else:
            if(text):
                self.l_text.append(text)
    def get_final_img(self):
        row2img=[]
        for row,imgs in enumerate(self.imgs):
            img=concat_images_list(*imgs,vert=False)
            row2img.append(img)
        ret=concat_images_list(*row2img,vert=True)
        return ret
    def clear(self):
        self.imgs = []
        self.l_text = []
