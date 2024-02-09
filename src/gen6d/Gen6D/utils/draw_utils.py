import matplotlib

# matplotlib.use('Agg')

# from utils.base_utils import compute_relative_transformation, compute_F
from .base_utils import compute_relative_transformation, compute_F

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm


def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
        ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

    l = mlines.Line2D([xmin, xmax], [ymin, ymax])
    ax.add_line(l)
    return l


def draw_correspondence(img0, img1, kps0, kps1, matches=None, colors=None, max_draw_line_num=None,
                        kps_color=(0, 0, 255), vert=False):
    if len(img0.shape) == 2:
        img0 = np.repeat(img0[:, :, None], 3, 2)
    if len(img1.shape) == 2:
        img1 = np.repeat(img1[:, :, None], 3, 2)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    if matches is None:
        assert (kps0.shape[0] == kps1.shape[0])
        matches = np.repeat(np.arange(kps0.shape[0])[:, None], 2, 1)

    if vert:
        w = max(w0, w1)
        h = h0 + h1
        out_img = np.zeros([h, w, 3], np.uint8)
        out_img[:h0, :w0] = img0
        out_img[h0:, :w1] = img1
    else:
        h = max(h0, h1)
        w = w0 + w1
        out_img = np.zeros([h, w, 3], np.uint8)
        out_img[:h0, :w0] = img0
        out_img[:h1, w0:] = img1

    for pt in kps0:
        pt = np.round(pt).astype(np.int32)
        cv2.circle(out_img, tuple(pt), 1, kps_color, -1)

    for pt in kps1:
        pt = np.round(pt).astype(np.int32)
        pt = pt.copy()
        if vert:
            pt[1] += h0
        else:
            pt[0] += w0
        cv2.circle(out_img, tuple(pt), 1, kps_color, -1)

    if max_draw_line_num is not None and matches.shape[0] > max_draw_line_num:
        np.random.seed(6033)
        idxs = np.arange(matches.shape[0])
        np.random.shuffle(idxs)
        idxs = idxs[:max_draw_line_num]
        matches = matches[idxs]

        if colors is not None and (type(colors) == list or type(colors) == np.ndarray):
            colors = np.asarray(colors)
            colors = colors[idxs]

    for mi, m in enumerate(matches):
        pt = np.round(kps0[m[0]]).astype(np.int32)
        pr_pt = np.round(kps1[m[1]]).astype(np.int32)
        if vert:
            pr_pt[1] += h0
        else:
            pr_pt[0] += w0
        if colors is None:
            cv2.line(out_img, tuple(pt), tuple(pr_pt), (0, 255, 0), 1)
        elif type(colors) == list or type(colors) == np.ndarray:
            color = (int(c) for c in colors[mi])
            cv2.line(out_img, tuple(pt), tuple(pr_pt), tuple(color), 1)
        else:
            color = (int(c) for c in colors)
            cv2.line(out_img, tuple(pt), tuple(pr_pt), tuple(color), 1)

    return out_img


def draw_keypoints(img, kps, colors=None, radius=2):
    out_img = img.copy()
    for pi, pt in enumerate(kps):
        pt = np.round(pt).astype(np.int32)
        if colors is not None:
            color = [int(c) for c in colors[pi]]
            cv2.circle(out_img, tuple(pt), radius, color, -1)
        else:
            cv2.circle(out_img, tuple(pt), radius, (0, 255, 0), -1)
    return out_img


def draw_epipolar_line(F, img0, img1, pt0, color):
    h1, w1 = img1.shape[:2]
    hpt = np.asarray([pt0[0], pt0[1], 1], dtype=np.float32)[:, None]
    l = F @ hpt
    l = l[:, 0]
    a, b, c = l[0], l[1], l[2]
    pt1 = np.asarray([0, -c / b]).astype(np.int32)
    pt2 = np.asarray([w1, (-a * w1 - c) / b]).astype(np.int32)

    img0 = cv2.circle(img0, tuple(pt0.astype(np.int32)), 5, color, 2)
    img1 = cv2.line(img1, tuple(pt1), tuple(pt2), color, 2)
    return img0, img1


def draw_epipolar_lines(F, img0, img1, num=20):
    img0, img1 = img0.copy(), img1.copy()
    h0, w0, _ = img0.shape
    h1, w1, _ = img1.shape

    for k in range(num):
        color = np.random.randint(0, 255, [3], dtype=np.int32)
        color = [int(c) for c in color]
        pt = np.random.uniform(0, 1, 2)
        pt[0] *= w0
        pt[1] *= h0
        pt = pt.astype(np.int32)
        img0, img1 = draw_epipolar_line(F, img0, img1, pt, color)

    return img0, img1


def gen_color_map(error, clip_max=12.0, clip_min=2.0):
    rectified_error = (error - clip_min) / (clip_max - clip_min)
    rectified_error[rectified_error < 0] = 0
    rectified_error[rectified_error >= 1.0] = 1.0
    viridis = cm.get_cmap('viridis', 256)
    colors = [viridis(e) for e in rectified_error]
    return np.asarray(np.asarray(colors)[:, :3] * 255, np.uint8)


def scale_float_image(image):
    max_val, min_val = np.max(image), np.min(image)
    image = (image - min_val) / (max_val - min_val) * 255
    return image.astype(np.uint8)


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


def concat_images_list(*args, vert=False):
    if len(args) == 1: return args[0]
    img_out = args[0]
    for img in args[1:]:
        img_out = concat_images(img_out, img, vert)
    return img_out


def get_colors_gt_pr(gt, pr=None):
    if pr is None:
        pr = np.ones_like(gt)
    colors = np.zeros([gt.shape[0], 3], np.uint8)
    colors[gt & pr] = np.asarray([0, 255, 0])[None, :]  # tp
    colors[(~gt) & pr] = np.asarray([255, 0, 0])[None, :]  # fp
    colors[gt & (~pr)] = np.asarray([0, 0, 255])[None, :]  # fn
    return colors


def draw_hist(fn, vals, bins=100, hist_range=None, names=None):
    if type(vals) == list:
        val_num = len(vals)
        if hist_range is None:
            hist_range = (np.min(vals), np.max(vals))
        if names is None:
            names = [str(k) for k in range(val_num)]
        for k in range(val_num):
            plt.hist(vals[k], bins=bins, range=hist_range, alpha=0.5, label=names[k])
        plt.legend()
    else:
        if hist_range is None:
            hist_range = (np.min(vals), np.max(vals))
        plt.hist(vals, bins=bins, range=hist_range)

    plt.savefig(fn)
    plt.close()


def draw_pr_curve(fn, gt_sort):
    pos_num_all = np.sum(gt_sort)
    pos_nums = np.cumsum(gt_sort)
    sample_nums = np.arange(gt_sort.shape[0]) + 1
    precisions = pos_nums.astype(np.float64) / sample_nums
    recalls = pos_nums / pos_num_all

    precisions = precisions[np.arange(0, gt_sort.shape[0], gt_sort.shape[0] // 40)]
    recalls = recalls[np.arange(0, gt_sort.shape[0], gt_sort.shape[0] // 40)]
    plt.plot(recalls, precisions, 'r-')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(fn)
    plt.close()


def draw_points(img, points):
    pts = np.round(points).astype(np.int32)
    h, w, _ = img.shape
    pts[:, 0] = np.clip(pts[:, 0], a_min=0, a_max=w - 1)
    pts[:, 1] = np.clip(pts[:, 1], a_min=0, a_max=h - 1)
    img = img.copy()
    img[pts[:, 1], pts[:, 0]] = 255
    # img[pts[:,1],pts[:,0]]+=np.asarray([127,0,0],np.uint8)[None,:]
    return img


from image_util import draw_bbox


def output_points(fn, pts, colors=None):
    with open(fn, 'w') as f:
        for pi, pt in enumerate(pts):
            f.write(f'{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} ')
            if colors is not None:
                f.write(f'{int(colors[pi, 0])} {int(colors[pi, 1])} {int(colors[pi, 2])}')
            f.write('\n')


def compute_axis_points(pose):
    R = pose[:, :3]  # 3,3
    t = pose[:, 3:]  # 3,1
    pts = np.concatenate([np.identity(3), np.zeros([3, 1])], 1)  # 3,4
    pts = R.T @ (pts - t)
    colors = np.asarray([[255, 0, 0], [0, 255, 0, ], [0, 0, 255], [0, 0, 0]], np.uint8)
    return pts.T, colors


def draw_epipolar_lines_func(img0, img1, Rt0, Rt1, K0, K1):
    Rt = compute_relative_transformation(Rt0, Rt1)
    F = compute_F(K0, K1, Rt[:, :3], Rt[:, 3:])
    return concat_images_list(*draw_epipolar_lines(F, img0, img1))


def pts_range_to_bbox_pts(max_pt, min_pt):
    maxx, maxy, maxz = max_pt
    minx, miny, minz = min_pt
    pts = [
        [minx, miny, minz],
        [minx, maxy, minz],
        [maxx, maxy, minz],
        [maxx, miny, minz],

        [minx, miny, maxz],
        [minx, maxy, maxz],
        [maxx, maxy, maxz],
        [maxx, miny, maxz],
    ]
    return np.asarray(pts, np.float32)


def draw_bbox_3d(img, pts2d, color=(0, 255, 0), thickness=2):
    red_colors = np.zeros([8, 3], np.uint8)
    red_colors[:, 0] = 255
    img = draw_keypoints(img, pts2d, colors=red_colors)

    pts2d = np.round(pts2d).astype(np.int32)
    img = cv2.line(img, tuple(pts2d[0]), tuple(pts2d[1]), color, thickness)
    img = cv2.line(img, tuple(pts2d[1]), tuple(pts2d[2]), color, thickness)
    img = cv2.line(img, tuple(pts2d[2]), tuple(pts2d[3]), color, thickness)
    img = cv2.line(img, tuple(pts2d[3]), tuple(pts2d[0]), color, thickness)

    img = cv2.line(img, tuple(pts2d[4]), tuple(pts2d[5]), color, thickness)
    img = cv2.line(img, tuple(pts2d[5]), tuple(pts2d[6]), color, thickness)
    img = cv2.line(img, tuple(pts2d[6]), tuple(pts2d[7]), color, thickness)
    img = cv2.line(img, tuple(pts2d[7]), tuple(pts2d[4]), color, thickness)

    img = cv2.line(img, tuple(pts2d[0]), tuple(pts2d[4]), color, thickness)
    img = cv2.line(img, tuple(pts2d[1]), tuple(pts2d[5]), color, thickness)
    img = cv2.line(img, tuple(pts2d[2]), tuple(pts2d[6]), color, thickness)
    img = cv2.line(img, tuple(pts2d[3]), tuple(pts2d[7]), color, thickness)
    return img


def draw_gradient_line(img, start_point, end_point, start_color, end_color, half_thickness):
    # print(start_color, end_color)
    """
    绘制颜色渐变的线段
    :param img: 图像
    :param start_point: 起始点坐标 (x1, y1)
    :param end_point: 终点坐标 (x2, y2)
    :param start_color: 起始颜色  (RGB)(0-1
    :param end_color: 终点颜色
    :param half_thickness:  int
    """
    
    length = int(np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2))
    angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])

    
    color_step = [(end_color[i] - start_color[i]) / length for i in range(3)]

    
    for i in range(length):
        
        current_color = [(start_color[j] + color_step[j] * i) for j in range(3)]
        current_color = tuple([int(x) for x in current_color])

        
        x = int(start_point[0] + i * np.cos(angle))
        y = int(start_point[1] + i * np.sin(angle))
        current_point = (x, y)

        
        cv2.circle(img, current_point, half_thickness, current_color, -1)


def draw_bbox_3d_dpt(img, pts2d, dpt, color=(0, 255, 0), thickness=2):
    """
    """
    # dpt = np.round(dpt).astype(np.int32)
    max_dpt = np.max(dpt)
    min_dpt = np.min(dpt)
    dpt_range = max_dpt - min_dpt
    colors = np.zeros([8, 3])  

    # colors[:, 0] = 255 * (1 - (dpt - min_dpt) / dpt_range)

    # colors[:, 0] = color[0] * (1 - (dpt - min_dpt) / dpt_range)
    # colors[:, 1] = color[1] * (1 - (dpt - min_dpt) / dpt_range)
    # colors[:, 2] = color[2] * (1 - (dpt - min_dpt) / dpt_range)
    def _map(x):
        _N=2
        y=255-x
        y=y* (1 - (dpt - min_dpt) / (dpt_range*_N))
        y=255-y
        return y
    colors[:, 0] = _map(color[0])
    colors[:, 1] = _map(color[1])
    colors[:, 2] = _map(color[2])

    # print(colors)
    img = draw_keypoints(img, pts2d, colors=colors)

    pts2d = np.round(pts2d).astype(np.int32)

    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        p1 = pts2d[i]
        p2 = pts2d[j]
        draw_gradient_line(img, p1, p2, colors[i], colors[j], half_thickness=thickness // 2)
    return img


if (__name__ == "__main__"):
    matplotlib.use('TkAgg')
    
    image = np.zeros((300, 400, 3))
    image = image + 150

    result_image = draw_bbox_3d_dpt(
        image, [
            
            [0, 0],
            [0, 100],
            [100, 100],
            [100, 0],
            [0, 200],
            [200, 100],
            [100, 200],
            [200, 0],
        ],
        [1, 2,
         # 3, 4, 5, 6, 7, 8
         1, 1, 8, 8, 8, 8
         ],
        (255, 0, 255)
    )

    
    result_image = result_image / 256
    print(result_image)
    plt.imshow(result_image)
    plt.axis('off')
    plt.show()
