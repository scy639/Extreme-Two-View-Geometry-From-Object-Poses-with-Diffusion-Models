from reportlab.pdfgen import canvas as reportlab_canvas
import cv2
from pathlib import Path
class PDF_A:
    """
    image as bg. add text on it
    """

    def __init__(self, save_path, ):
        self.save_path=save_path
        self.canvas=None
        self.__historyImagePath=[]



    def add_text(self, text, x, y, font_size, color='#000000'):
        """
        x, y, font_size: pixel or relative (for font_size, relative to height)
        """
        def process(a, abs_val, min_, max_):
            def is_relative(a):
                if isinstance(a, float):
                    if 0 <= a <= 1:
                        return True
                    else:
                        assert 0, a
                else:
                    return False

            if is_relative(a):
                a = abs_val * a
                a = int(a)
                a = min(a, max_)
                a = max(a, min_)
            return a

        x = process(x, abs_val=self.width, min_=0, max_=self.width)
        y = process(y, abs_val=self.height, min_=0, max_=self.height)
        font_size = process(font_size, abs_val=self.height, min_=0, max_=self.height)

        c = self.canvas
        FONT_SIZE = font_size
        y = self.height - FONT_SIZE - y
        c.setFont("Helvetica", FONT_SIZE)
        c.setFillColor(color)
        c.drawString(x, y, text)

    def new_page(self, image_path):
        """
             new page.   image as bg of this page
        """
        assert image_path not in self.__historyImagePath,'reportlab的傻逼机制，如果之前已经有了，它就复用之前的，不会重新读'
        image = cv2.imread(image_path)
        width = image.shape[1]
        height = image.shape[0]
        pagesize = (width, height)
        if self.canvas is None:
            # Create an empty PDF
            self.canvas = reportlab_canvas.Canvas(self.save_path, pagesize=None)
        else:
            self.canvas.showPage()
        self.canvas.setPageSize(pagesize)
        self.canvas.drawImage(image_path, 0, 0)
        self.width = width
        self.height = height
        self.__historyImagePath.append(image_path)

    def save(self):
        print(f"[PDF_A.save] saving...")
        self.canvas.save()

# from pdf_util import PDF_A
from vis.cv2_util import putText_B, concat_images_list
import os
import cv2

def gen_overview_of_images(
        #input
        imgPaths_or_folder,
        #output
        pdf_path=None,
        overviewImg_path=None,
        #
        interval=1,
        #
        num_per_row=6,
        row_per_page=5,
        #
        max_size=None,
        #
        text_color=(20,250,20),
        fontScale=1.5,
        callback_path2text=lambda imgPath: os.path.splitext(os.path.basename(imgPath))[0],
        #
        padding_color='not used',
        tmp_img_format__if_no_overviewImg_path='jpg',
):
    """
    imgPaths_or_folder: if list, then it's imgPaths, if str, it's input folder, then imgPaths=all images in this folder
    pdf_path: output pdf path
    interval: sample interval of imgPaths
    num_per_row: how many image in a row
    row_per_page: max row per page
    max_size: max h or w for each image
    font_color:
    fontScale:
    padding_color: padding color between images
    """
    assert overviewImg_path or pdf_path,'至少得保存一种吧'
    if isinstance(imgPaths_or_folder, Path):
        imgPaths_or_folder=str(imgPaths_or_folder)
    # Determine if imgPaths_or_folder is a list of image paths or a folder path
    if isinstance(imgPaths_or_folder, str):
        imgPaths = [os.path.join(imgPaths_or_folder, f) for f in os.listdir(imgPaths_or_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    elif isinstance(imgPaths_or_folder, list):
        imgPaths = imgPaths_or_folder
    else:
        raise ValueError("imgPaths_or_folder should be a folder path or a list of image paths")

    pdfA = PDF_A(pdf_path)

    assert interval>=1
    imgPaths = imgPaths[::interval]
    def save(l_row_image,tmp_img_path):
        page_image = concat_images_list(*l_row_image, vert=True)
        # Add the row image to the PDF
        cv2.imwrite(tmp_img_path, page_image)
        pdfA.new_page(tmp_img_path)
    def get_img_path(i):
        if overviewImg_path:
            ret=overviewImg_path
            if i>0:
                #'...(i).png/jpg'
                ret=Path(ret)
                ret=ret.parent/(ret.stem+f"({i})"+ret.suffix)
                ret=str(ret)
            abs_path=os.path.abspath(ret)
            print(f"save overview img to {abs_path}")
        else:
            ret=f"tmp4[gen_overview_of_images]{i}.{tmp_img_format__if_no_overviewImg_path}"
        return ret
    l_row_image=[]
    for i in range(0,len(imgPaths),num_per_row ):
        # Get the subset of images for each row
        imgPaths_subset = imgPaths[i:i+num_per_row]

        # Create the row of images
        row_images = [cv2.imread(imgPath) for imgPath in imgPaths_subset]
        if max_size is not None:
            for j,img in enumerate(row_images):
                # Resize the images based on the max_size
                # ttt_w = min(img.shape[1], max_size)
                # ttt_h = min(img.shape[0], max_size)
                # ttt_h = ttt_w * img.shape[0] // img.shape[1]
                # row_images[j] = cv2.resize(img, (ttt_w, ttt_h))
                row_images[j] = cv2.resize(img, max_size)
        # Add text to each image
        for j, img in enumerate(row_images):
            """
            if isinstance(text_size,float):
                assert 0<=text_size<=1
                _text_size=min(img.shape[0],img.shape[1])
                _text_size=int(   _text_size*text_size   )
            elif isinstance(text_size,int):
                _text_size=text_size
            else:
                raise ValueError
            """
            # row_images[j] = putText_B(
            img= putText_B(
                img,
                callback_path2text(imgPaths_subset[j]),
                org=(5, 5),
                thickness=1,
                fontScale=fontScale, color=text_color,
            )

        # Concatenate the row of images
        row_image = concat_images_list(*row_images, vert=False)
        l_row_image.append(row_image)
        
        if len(l_row_image)==row_per_page:
            save(l_row_image,
                 tmp_img_path = get_img_path(i))
            l_row_image = []
    if l_row_image!=[]:
        save(l_row_image,
                 tmp_img_path = get_img_path(i))
        l_row_image = []
    if pdf_path: 
        pdfA.save()

if(__name__=="__main__"):
    import glob
    gen_overview_of_images(
        imgPaths_or_folder=glob.glob(r"G:\4renderGSO\output\gso*\*.png")[:400],
        pdf_path='./gen_overview_of_images.pdf',
        interval=10,
        row_per_page=3,
    )
    exit(0)
if __name__ == "__main__":
    pdfA = PDF_A("PDF_A.pdf", )
    pdfA.new_page("0.png")
    pdfA.add_text("Hello, World!", 0, 0, 12)
    pdfA.add_text("Hello, World 2!", 0.1, 0.2, 0.2)
    pdfA.add_text("Hello, World 2!", 0.1, 0, 0.1, color=(255, 0, 0))
    pdfA.add_text("Hello, World 2!", 0, 0, 0.1, color='#00ff00')
    pdfA.new_page("0.jpg")  # Add a new page with a different image
    pdfA.add_text("Hello, World on Page 2!", 0, 0, 12)
    pdfA.save()