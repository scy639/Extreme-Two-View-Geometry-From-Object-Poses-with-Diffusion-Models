"""
borrow from relposepp demo.py. plotly_scene_visualization is from colab( different from demo.py)
"""

import argparse
import base64
import io
import json
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly
import torch
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene



def plotly_scene_visualization(R_pred, T_pred, name="",colors=None):
    # Construct cameras and visualize scene for quick solution
    cameras_pred = FoVPerspectiveCameras(R=R_pred, T=T_pred)
    scenes = {name: {}}
    num_frames = R_pred.shape[0]
    for i in range(num_frames):
        scenes[name][i] = FoVPerspectiveCameras(R=R_pred[i, None], T=T_pred[i, None])

    fig = plot_scene(
        scenes,
        camera_scale=0.03,
        ncols=2,
    )
    fig.update_scenes(aspectmode="data")

    cmap = plt.get_cmap("hsv")
    for i in range(num_frames):
        fig.data[i].line.color = matplotlib.colors.to_hex(cmap(i / (num_frames)))
        if(colors!=None):
            fig.data[i].line.color =colors[i]
    fig.show()

if(__name__ == '__main__'):
    # pose_save_path_format=rf"C:\Users\YiLucky\Desktop\[gen6d-4]hydrant-106_12648_23157-44,80 {{}}.npy"
    pose_save_path_format=rf"C:\Users\YiLucky\Desktop\[gen6d-4]hydrant-106_12648_23157-68,55 {{}}.npy"
    # pose_save_path_format=rf"C:\Users\YiLucky\Desktop\[gen6d-4]hydrant-106_12648_23157-78,93 {{}}.npy"
    # pose_save_path_format=rf"C:\Users\YiLucky\Desktop\[gen6d-4]hydrant-106_12648_23157-95,65 {{}}.npy"

    base_w2c = np.eye(4)
    w2c=np.load(pose_save_path_format.format("w2c"))
    gt_w2c=np.load(pose_save_path_format.format("gt_w2c"))
    print(w2c,gt_w2c,w2c @ gt_w2c,sep="\n")

    from vis_rel_pose import *
    Rs=np.stack( pose[:3,:3] for pose in  [w2c,gt_w2c,base_w2c])
    ts=np.stack([look_at_origin_R2t_w2cColumn(R,distance=0.5) for R in Rs])
    fig = plotly_scene_visualization(Rs, ts ,colors=["#0000ff","#ffff00","grey"])
    fig.show()





    # html_plot = plotly.io.to_html(fig, full_html=False, include_plotlyjs="cdn")
    # s = io.BytesIO()
    # view_color_coded_images_from_tensor(images)
    # plt.savefig(s, format="png", bbox_inches="tight")
    # plt.close()
    # image_encoded = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    # with open(output_path, "w") as f:
    #     s = HTML_TEMPLATE.format(
    #         image_encoded=image_encoded,
    #         plotly_html=html_plot,
    #     )
    #     f.write(s)
