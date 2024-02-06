import  root_config,time
# from imports import *
from exception_util import handle_exception

def main(read_path, save_path,ask):
    SRC = read_path
    new_path = save_path
    """
    """
    import os
    import cv2

    # SRC = "original-png-4samples"
    # SRC="test"

    def get_match_score(img0, img1) -> float:
        """
        :param img0: 0-0(x=0,y=30.0,z=0).png
        :param img1: 0-1(x=0,y=30.0,z=0).png
        :return: match score

        """

        """
        # copilot version:
            # :logic:  1. get sift feature of img0 and img1
            #      2. match sift feature
            #      3. get match score
        sift = cv2.xfeatures2d.SIFT_create()
        kp0, des0 = sift.detectAndCompute(img0, None)
        kp1, des1 = sift.detectAndCompute(img1, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des0, des1, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append([m])
        return len(good)
        """

        """
        # conv version
        # calculate conv( matrix mul and accumulate all elements

        return (img0 * img1).sum()
        """

        """
        计算每个ele之间的距离然后sum
        """
        # return -(img0 - img1).sum()
        return -((img0 - img1) ** 2).sum()

    def choose_j(pre_img, l_cur_img: list):
        """
        :param pre_img: i-1时的最终被选图片
        :param l_cur_img: i时所有备选图片
        :return: 最终被选图片的index(j
        :logic:  选择与pre_img最match的.
        """

        score = 0
        j = 0
        for i in range(len(l_cur_img)):
            cur_img = l_cur_img[i]
            cur_score = get_match_score(pre_img, cur_img)
            if (cur_score > score):
                score = cur_score
                j = i
        return j

    # s1: get i2samples
    files = os.listdir(SRC)
    i2samples = {}
    for file in files:
        # if is jpg
        if (file.split('.')[-1] != 'jpg'):
            continue
        # file=11-2(x=0,y=30.0,z=0).png=i-j(x=0,y=30.0,z=0).png
        i = int(file.split('-')[0])
        j = int(file.split('-')[1].split('(')[0])  # index of sample
        rest = file[len(f"{i}-j"):]
        if (i not in i2samples):
            i2samples[i] = []
        i2samples[i].append(file)

    # s2: i2samples 2 l_copy
    def file2img(file: str):
        return cv2.imread(f"{SRC}/{file}")

    l_copy = []
    pre_img = None
    for i in range(len(i2samples)):
        # for i in [44,45]:
        cur_fileNames = i2samples[i]
        if(root_config.USE_ALL_SAMPLE):
            for j in range(root_config.NUM_SAMPLE):
                l_copy.append((
                    cur_fileNames[j],
                    # f"{i}.jpg"
                    cur_fileNames[j]
                ))
        else:
            if (pre_img is None):
                pre_img = file2img(cur_fileNames[0])
            # cur_fileNames 2 cur_imgs. read img to nd array
            cur_imgs = [file2img(cur_fileName) for cur_fileName in cur_fileNames]
            j = choose_j(pre_img, cur_imgs)
            pre_img = cur_imgs[j]
            l_copy.append((
                cur_fileNames[j],
                # f"{i}.jpg"
                cur_fileNames[j]
            ))

    # print(l_copy)
    import shutil

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for copy in l_copy:
        # new_path = f"onlyI-png-1samples"
        try:
            shutil.copy(f"{SRC}/{copy[0]}", f"{new_path}/{copy[1]}")
        except BlockingIOError as e:
            time.sleep(1)
            pERROR(f'--------BlockingIOError------shutil.copy({f"{SRC}/{copy[0]}", f"{new_path}/{copy[1]}"}-----')
            handle_exception(e)
