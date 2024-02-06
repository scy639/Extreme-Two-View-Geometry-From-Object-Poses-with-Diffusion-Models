# 
import glob
from debug_util import debug_imsave
from misc_util import your_datetime,get_datetime_str
CHECK_NSFW = False

'''
conda activate zero123
cd stable-diffusion
python gradio_new.py 0
'''
from imports import *
import diffusers  # 0.12.1
import os
import math
import fire
from image_util import print_image_statistics
# with HiddenSpecified_OutAndErr(['DeprecationWarning: jsonschema.RefResolver is deprecated']):
#     # print('import gradio, enter HiddenSpecified_OutAndErr')
#     import gradio as gr
import lovely_numpy
import lovely_tensors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import rich
import sys
import time
import torch
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from functools import partial
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from ldm.util import add_margin
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
# from rich import print
from transformers import AutoFeatureExtractor  # , CLIPImageProcessor
from torch import autocast
from torchvision import transforms
import gpu_util 
_SHOW_DESC = True
_SHOW_INTERMEDIATE = False
# _SHOW_INTERMEDIATE = True
_GPU_INDEX = root_config.GPU_INDEX
# _GPU_INDEX = 2

# _TITLE = 'Zero-Shot Control of Camera Viewpoints within a Single Image'
_TITLE = 'Zero-1-to-3: Zero-shot One Image to 3D Object'
CONFIG='configs/sd-objaverse-finetune-c_concat-256.yaml'
device = f'cuda:{root_config.GPU_INDEX}'
# This demo allows you to generate novel viewpoints of an object depicted in an input image using a fine-tuned version of Stable Diffusion.
_DESCRIPTION = '''
This demo allows you to control camera rotation and thereby generate novel viewpoints of an object within a single image.
It is based on Stable Diffusion. Check out our [project webpage](https://zero123.cs.columbia.edu/) and [paper](https://arxiv.org/) if you want to learn more about the method!
Note that this model is not intended for images of humans or faces, and is unlikely to work well for them.
'''

_ARTICLE = 'See uses.md'


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    # gpu_util.occumpy_mem_for_ckpt( weight_path=ckpt )
    if root_config.LOAD_BY_IPC:
        from miscellaneous.MemoryCache import MemoryCache
        import io
        buffer=MemoryCache.receive()
        buffer = io.BytesIO(buffer)
        pl_sd = torch.load(buffer, map_location='cpu'  )
        del buffer
    else:
        pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    # import sys
    # print("e2vg_run1 sys.path:", sys.path)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(input_im, 
                 model,#:ldm.models.diffusion.ddpm.LatentDiffusion, 
                 sampler:DDIMSampler,
                 precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z):
    """ 
    model是models['turncam']
    """
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope(root_config.DEVICE):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


@torch.no_grad()  
def sample_model_batch(model, sampler:DDIMSampler, input_im, xs, ys, zs, n_samples=4, precision='autocast', ddim_eta=1.0,
                       ddim_steps=75, scale=3.0, h=256, w=256):
    assert n_samples==len(xs)==len(ys)==len(zs)
    print(f"n_samples,len(xs)= {n_samples,len(xs) }")
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope(root_config.DEVICE):
        # with model.ema_scope():
        c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
        T = []
        for x, y, z in zip(xs, ys, zs):
            T.append([np.radians(x), np.sin(np.radians(y)), np.cos(np.radians(y)), z])
        T = torch.tensor(np.array(T))[:, None, :].float().to(c.device)
        c = torch.cat([c, T], dim=-1)
        c = model.cc_projection(c)
        cond = {}
        cond['c_crossattn'] = [c]
        cond['c_concat'] = [model.encode_first_stage(input_im).mode().detach()
                            .repeat(n_samples, 1, 1, 1)]
        if scale != 1.0:
            uc = {}
            uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
            uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
        else:
            uc = None

        shape = [4, h // 8, w // 8]
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            x_T=None)
        # print(samples_ddim.shape)
        # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
        x_samples_ddim = model.decode_first_stage(samples_ddim)# BS, 3, 256, 256
        ret_imgs = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
        del cond, c, x_samples_ddim, samples_ddim, uc, input_im
        torch.cuda.empty_cache()
        return ret_imgs

from miscellaneous.Zero123_BatchB_Input import  Zero123_BatchB_Input
@torch.no_grad()
def sample_model_batchB(
        model, sampler:DDIMSampler,
        l__zero123_BatchB_Input:list[Zero123_BatchB_Input],
        # precision='autocast', ddim_eta=1.0, ddim_steps=75, scale=3.0, h=256, w=256,
        precision, ddim_eta, ddim_steps, scale, h, w,
):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope(root_config.DEVICE):
        cond = {}
        # ----cond['c_concat']---------
        l_c_concat=[]
        sum_n_samples=0
        for zero123_BatchB_Input in l__zero123_BatchB_Input:
            _, input_im, l_xyz = zero123_BatchB_Input.folder_outputIms, zero123_BatchB_Input.input_image_path, zero123_BatchB_Input.l_xyz
            _n_samples=len(l_xyz)
            sum_n_samples+=_n_samples
            l_c_concat.append( model.encode_first_stage(input_im).mode().detach()
                            .repeat(_n_samples, 1, 1, 1)  )
        l_c_concat = torch.concat(l_c_concat,dim=0)
        assert l_c_concat.shape[0]==sum_n_samples
        cond['c_concat']=[  l_c_concat  ]
        # ----c---------
        l_c = []
        for zero123_BatchB_Input in l__zero123_BatchB_Input:
            _, input_im, l_xyz = zero123_BatchB_Input.folder_outputIms, zero123_BatchB_Input.input_image_path, zero123_BatchB_Input.l_xyz
            _n_samples = len(l_xyz)
            l_c.append(  model.get_learned_conditioning(input_im).tile(_n_samples, 1, 1)  )
        c = torch.concat(l_c, dim=0)
        assert c.shape[0] == sum_n_samples
        # -----T-------
        T = []
        for zero123_BatchB_Input in l__zero123_BatchB_Input:
            _, _, l_xyz = zero123_BatchB_Input.folder_outputIms, zero123_BatchB_Input.input_image_path, zero123_BatchB_Input.l_xyz
            for x, y, z in l_xyz:
                T.append([np.radians(x), np.sin(np.radians(y)), np.cos(np.radians(y)), z])
        assert len(T) == sum_n_samples
        #------------
        T = torch.tensor(np.array(T))[:, None, :].float().to(c.device)
        c = torch.cat([c, T], dim=-1)
        c = model.cc_projection(c)
        cond['c_crossattn'] = [c]
        # ------uc------
        if scale != 1.0:
            uc = {}
            uc['c_concat'] = [torch.zeros(sum_n_samples, 4, h // 8, w // 8).to(c.device)]
            uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
        else:
            uc = None
        # ------------
        shape = [4, h // 8, w // 8]
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=sum_n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            x_T=None)
        # print(samples_ddim.shape)
        # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
        x_samples_ddim = model.decode_first_stage(samples_ddim)# BS, 3, 256, 256
        ret_imgs = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
        del cond, c, x_samples_ddim, samples_ddim, uc, input_im
        torch.cuda.empty_cache()
        return ret_imgs

def sample_model_batchB_wrapper(
        l__zero123_BatchB_Input:list[Zero123_BatchB_Input],
        #------------------
        scale=3.0,
        #  scale=20.0,
        ddim_steps=75,
        ddim_eta=1.0,
        precision='fp32',
        h=256, w=256,
):
    #TODO check exists
    if len(l__zero123_BatchB_Input)==0:
        return
    all_output_image_path=[]#for check
    new_l__zero123_BatchB_Input=[]
    for zero123_BatchB_Input in l__zero123_BatchB_Input:
        id_,folder_outputIms,input_image_path,l_xyz=zero123_BatchB_Input.id_,zero123_BatchB_Input.folder_outputIms,zero123_BatchB_Input.input_image_path,zero123_BatchB_Input.l_xyz
        if 1:
            assert '/' not in folder_outputIms
            assert isinstance(folder_outputIms,str)
            assert len(folder_outputIms)>0
            assert folder_outputIms.startswith('_'),"建议startswith('_')以示区分"
            path_output_ims = os.path.join(root_config.dataPath_zero123, folder_outputIms,id_)
            del folder_outputIms
        else:
            path_output_ims = os.path.join(root_config.dataPath_zero123, id_)
        if not os.path.exists(path_output_ims):
            os.makedirs(path_output_ims)
        raw_im = Image.open(input_image_path)
        print("zero123_BatchB_Input.input_image_path= ", input_image_path)
        # ---------check exist
        new_xyzLinearGradient = []
        zero123_BatchB_Input.outputims=[]
        for i, (x, y, z) in enumerate(l_xyz):
            img_save_path = os.path.join(path_output_ims, ImagePathUtil.get_path(i, 0, x, y, z, ))
            zero123_BatchB_Input.outputims.append(img_save_path)
            if not os.path.exists(img_save_path) or root_config.FORCE_zero123_render_even_img_exist:
                new_xyzLinearGradient.append((x, y, z))
                ttt345 = glob.glob(os.path.join(path_output_ims, ImagePathUtil.i2glob(i)))
                assert ttt345 == [], f"不应有 other img ({ttt345}) that has the same i:{i}"
                all_output_image_path.append(img_save_path)
        if (len(new_xyzLinearGradient) == 0):
            print(f"[sample_model_batchB_wrapper] all images already exist, eg. {img_save_path}", )
            continue
        else:
            assert len(new_xyzLinearGradient) == len(
                l_xyz), f"目前output_ims是全部gen完才保存，故要么new_xyzLinearGradient==[],要么new_xyzLinearGradient与xyzLinearGradient一样。new_xyzLinearGradient= {new_xyzLinearGradient}\nl_xyz= {l_xyz}"
        zero123_BatchB_Input.l_xyz = new_xyzLinearGradient
        del new_xyzLinearGradient
        #---------
        input_im = preprocess_image(None, raw_im, True)
        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
        input_im = input_im * 2 - 1
        input_im = transforms.functional.resize(input_im, [h, w])
        zero123_BatchB_Input.input_image_path=input_im
        zero123_BatchB_Input.folder_outputIms=path_output_ims
        new_l__zero123_BatchB_Input.append(zero123_BatchB_Input)
    total_len=sum( [len(i)for i in new_l__zero123_BatchB_Input]  )
    assert len(all_output_image_path)==total_len
    if len(new_l__zero123_BatchB_Input)==0:
        return l__zero123_BatchB_Input
    #-----------------------
    ModelsGetter.set_param(f'cuda:{root_config.GPU_INDEX}', root_config.weightPath_zero123, CONFIG)
    torch.cuda.empty_cache()  # 
    models = ModelsGetter.get_B()
    sampler = DDIMSampler(models['turncam'])
    sample_batch_B_size = root_config.SAMPLE_BATCH_B_SIZE
    if len(new_l__zero123_BatchB_Input) < sample_batch_B_size:
        sample_batch_B_size = len(new_l__zero123_BatchB_Input)
    num_batch = len(new_l__zero123_BatchB_Input) // sample_batch_B_size
    l = []
    end_i = None
    for i in range(num_batch):
        end_i = (i + 1) * sample_batch_B_size
        l.append(
            sample_model_batchB(
                models['turncam'], sampler,
                new_l__zero123_BatchB_Input[i * sample_batch_B_size:end_i],
                precision, ddim_eta, ddim_steps, scale, h, w,
            )
        )
    if len(new_l__zero123_BatchB_Input) % sample_batch_B_size != 0:
        l.append(
            sample_model_batchB(
                models['turncam'], sampler,
                new_l__zero123_BatchB_Input[end_i:],
                precision, ddim_eta, ddim_steps, scale, h, w,
            )
        )
    del sample_batch_B_size
    del end_i
    x_samples_ddim = []
    for i in range(len(l)):
        x_samples_ddim += l[i]
    assert len(x_samples_ddim) == total_len
    #---------
    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    output_paths = []#for check
    i_output_im=0
    for zero123_BatchB_Input in new_l__zero123_BatchB_Input:
        folder_outputIms,input_image_path,l_xyz=zero123_BatchB_Input.folder_outputIms,zero123_BatchB_Input.input_image_path,zero123_BatchB_Input.l_xyz
        tmp_outputims=[]
        for i, xyz in enumerate(l_xyz):
            x,y,z=xyz
            img_save_path = os.path.join(folder_outputIms, ImagePathUtil.get_path(i, 0, x, y, z, ))
            output_im=output_ims[i_output_im]
            i_output_im+=1
            output_im.save(img_save_path)
            print("has saved img: ", img_save_path)
            #
            output_paths.append(img_save_path)
            tmp_outputims.append(img_save_path)
        assert zero123_BatchB_Input.outputims==tmp_outputims
        assert len(zero123_BatchB_Input.outputims)==len(zero123_BatchB_Input)
    assert i_output_im==len(output_ims)
    ModelsGetter.release_gmem()
    assert output_paths==all_output_image_path,f"{output_paths},{all_output_image_path}"
    return l__zero123_BatchB_Input
class CameraVisualizer:
    def __init__(self, gradio_plot):
        self._gradio_plot = gradio_plot
        self._fig = None
        self._polar = 0.0
        self._azimuth = 0.0
        self._radius = 0.0
        self._raw_image = None
        self._8bit_image = None
        self._image_colorscale = None

    def polar_change(self, value):
        self._polar = value
        # return self.update_figure()

    def azimuth_change(self, value):
        self._azimuth = value
        # return self.update_figure()

    def radius_change(self, value):
        self._radius = value
        # return self.update_figure()

    def encode_image(self, raw_image):
        '''
        :param raw_image (H, W, 3) array of uint8 in [0, 255].
        '''
        # https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot

        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype='uint8')).convert('P', palette='WEB')
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

        self._raw_image = raw_image
        self._8bit_image = Image.fromarray(raw_image).convert('P', palette='WEB', dither=None)
        # self._8bit_image = Image.fromarray(raw_image.clip(0, 254)).convert(
        #     'P', palette='WEB', dither=None)
        self._image_colorscale = [
            [i / 255.0, 'rgb({}, {}, {})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]

        # return self.update_figure()

    def update_figure(self):
        fig = go.Figure()

        if self._raw_image is not None:
            (H, W, C) = self._raw_image.shape

            x = np.zeros((H, W))
            (y, z) = np.meshgrid(np.linspace(-1.0, 1.0, W), np.linspace(1.0, -1.0, H) * H / W)
            print('x:', lo(x))
            print('y:', lo(y))
            print('z:', lo(z))

            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                surfacecolor=self._8bit_image,
                cmin=0,
                cmax=255,
                colorscale=self._image_colorscale,
                showscale=False,
                lighting_diffuse=1.0,
                lighting_ambient=1.0,
                lighting_fresnel=1.0,
                lighting_roughness=1.0,
                lighting_specular=0.3))

            scene_bounds = 3.5
            base_radius = 2.5
            zoom_scale = 1.5  # Note that input radius offset is in [-0.5, 0.5].
            fov_deg = 50.0
            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

            input_cone = calc_cam_cone_pts_3d(
                0.0, 0.0, base_radius, fov_deg)  # (5, 3).
            output_cone = calc_cam_cone_pts_3d(
                self._polar, self._azimuth, base_radius + self._radius * zoom_scale, fov_deg)  # (5, 3).
            # print('input_cone:', lo(input_cone).v)
            # print('output_cone:', lo(output_cone).v)

            for (cone, clr, legend) in [(input_cone, 'green', 'Input view'),
                                        (output_cone, 'blue', 'Target view')]:

                for (i, edge) in enumerate(edges):
                    (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                    (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                    (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                    fig.add_trace(go.Scatter3d(
                        x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                        line=dict(color=clr, width=3),
                        name=legend, showlegend=(i == 0)))
                    # text=(legend if i == 0 else None),
                    # textposition='bottom center'))
                    # hoverinfo='text',
                    # hovertext='hovertext'))

                # Add label.
                if cone[0, 2] <= base_radius / 2.0:
                    fig.add_trace(go.Scatter3d(
                        x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] - 0.05], showlegend=False,
                        mode='text', text=legend, textposition='bottom center'))
                else:
                    fig.add_trace(go.Scatter3d(
                        x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] + 0.05], showlegend=False,
                        mode='text', text=legend, textposition='top center'))

            # look at center of scene
            fig.update_layout(
                # width=640,
                # height=480,
                # height=400,
                height=360,
                autosize=True,
                hovermode=False,
                margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                showlegend=True,
                legend=dict(
                    yanchor='bottom',
                    y=0.01,
                    xanchor='right',
                    x=0.99,
                ),
                scene=dict(
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=1.0),
                    camera=dict(
                        eye=dict(x=base_radius - 1.6, y=0.0, z=0.6),
                        center=dict(x=0.0, y=0.0, z=0.0),
                        up=dict(x=0.0, y=0.0, z=1.0)),
                    xaxis_title='',
                    yaxis_title='',
                    zaxis_title='',
                    xaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    yaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    zaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks='')))

        self._fig = fig
        return fig

NO_CARVEKIT=root_config.NO_CARVEKIT    #     zero123 no remove bg again (no_carvekit)
def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    if NO_CARVEKIT :   
        aspect=input_im.width/input_im.height
        assert 0.9<aspect<1.1,'可能会影响效果？比如resize后obj被压扁，影响zero123对它的理解'
        """input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0"""
        #no_carvekit_B
        input_im.thumbnail([220, 220], Image.Resampling.LANCZOS)
        input_im = add_margin(input_im, (255, 255, 255), size=256)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        assert input_im.shape[2]==3
        return input_im
    start_time = time.time()
    if preprocess:
        models=ModelsGetter.get_B()
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        print('after load_and_preprocess', input_im.shape)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    print('new input_im:', lo(input_im))

    return input_im


def main_run(models, device, cam_vis, return_what,
             x=0.0, y=0.0, z=0.0,
             raw_im=None, preprocess=True,
             scale=3.0, 
            #  scale=20.0, 
             n_samples=4,
             ddim_steps=50,
             #  ddim_steps=77,
             ddim_eta=1.0,
             precision='fp32', h=256, w=256,
             batch_sample=False,
             ):
    '''
    :param raw_im (PIL Image).

    x:Polar angle (vertical rotation in degrees)
    y:Azimuth angle (horizontal rotation in degrees)
    z:Zoom (relative distance from center)
    '''

    # raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)

    if CHECK_NSFW:  # 
        safety_checker_input = models['clip_fe'](raw_im, return_tensors='pt').to(device)
        (image, has_nsfw_concept) = models['nsfw'](
            images=np.ones((1, 3)), clip_input=safety_checker_input.pixel_values)
        print('has_nsfw_concept:', has_nsfw_concept)
        if np.any(has_nsfw_concept):
            print('NSFW content detected.')
            to_return = [None] * 10
            description = ('###  <span style="color:red"> Unfortunately, '
                           'potential NSFW content was detected, '
                           'which is not supported by our model. '
                           'Please try again with a different image. </span>')
            if 'angles' in return_what:
                to_return[0] = 0.0
                to_return[1] = 0.0
                to_return[2] = 0.0
                to_return[3] = description
            else:
                to_return[0] = description
            return to_return

        else:
            print('Safety check passed.')
    else:
        pass

    # debug_imsave(f'e2vg_run1-preprocess_image/{root_config.refIdSuffix}--{root_config.idSuffix}/{NO_CARVEKIT}-before--{get_datetime_str()}.jpg',raw_im)
    input_im = preprocess_image(models, raw_im, preprocess)
    # print_image_statistics(input_im)
    # debug_imsave(f'e2vg_run1-preprocess_image/{root_config.refIdSuffix}--{root_config.idSuffix}/{NO_CARVEKIT}-after--{get_datetime_str()}.jpg',input_im)
    # debug_imsave(f'debugNoCarvekit/{str(Path(raw_im.filename).name)}--{root_config.idSuffix}.png',input_im)
    
    
    
    
    # if np.random.rand() < 0.3:
    #     description = ('Unfortunately, a human, a face, or potential NSFW content was detected, '
    #                    'which is not supported by our model.')
    #     if vis_only:
    #         return (None, None, description)
    #     else:
    #         return (None, None, None, description)

    show_in_im1 = (input_im * 255.0).astype(np.uint8)
    show_in_im2 = Image.fromarray(show_in_im1)

    if 'rand' in return_what:
        x = int(np.round(np.arcsin(np.random.uniform(-1.0, 1.0)) * 160.0 / np.pi))  # [-80, 80].
        y = int(np.round(np.random.uniform(-150.0, 150.0)))
        z = 0.0
    if (cam_vis):
        cam_vis.polar_change(x)
        cam_vis.azimuth_change(y)
        cam_vis.radius_change(z)
        cam_vis.encode_image(show_in_im1)
        new_fig = cam_vis.update_figure()
    else:
        new_fig = None
    if 'vis' in return_what:
        description = ('The viewpoints are visualized on the top right. '
                       'Click Run Generation to update the results on the bottom right.')

        if 'angles' in return_what:
            return (x, y, z, description, new_fig, show_in_im2)
        else:
            return (description, new_fig, show_in_im2)

    elif 'gen' in return_what:
        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
        input_im = input_im * 2 - 1
        input_im = transforms.functional.resize(input_im, [h, w])
        models=ModelsGetter.get_B()
        torch.cuda.empty_cache()
        sampler = DDIMSampler(models['turncam'])
        if (batch_sample):
            assert isinstance(x, list) and isinstance(y, list) and isinstance(z, list)
            assert len(x) == len(y) and len(y) == len(z)
            sample_batch_size=root_config.SAMPLE_BATCH_SIZE
            # x_samples_ddim = sample_model_batch(models['turncam'], sampler, input_im, x, y, z, n_samples=len(x),
            #                                     precision=precision,
            #                                     ddim_steps=ddim_steps, scale=scale)
            if(len(x)<sample_batch_size):
                sample_batch_size=len(x)
            if len(x) % sample_batch_size == 0:
                num_batch=len(x)//sample_batch_size
            else:
                
                num_batch=len(x)//sample_batch_size
            l=[]
            end_i=None
            for i in range(num_batch):
                end_i=(i + 1) * sample_batch_size
                l.append(
                    sample_model_batch(
                        models['turncam'], sampler, input_im,
                        x[i * sample_batch_size:end_i],
                        y[i * sample_batch_size:end_i],
                        z[i * sample_batch_size:end_i],
                        n_samples=sample_batch_size,
                        precision=precision,
                        ddim_steps=ddim_steps, scale=scale
                    )
                )
            if len(x) % sample_batch_size != 0:
                sample_batch_size=len(x)-end_i
                l.append(
                    sample_model_batch(
                        models['turncam'], sampler, input_im,
                        x[end_i: ],
                        y[end_i: ],
                        z[end_i: ],
                        n_samples=sample_batch_size,
                        precision=precision,
                        ddim_steps=ddim_steps, scale=scale
                    )
                )
            del sample_batch_size
            del end_i
            x_samples_ddim=[]
            # for i in range(num_batch):bug
            for i in range(len(l)):
                x_samples_ddim+=l[i]
            assert len(x_samples_ddim)==len(x)
        else:
            # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
            used_x = x  # NOTE: Set this way for consistency.
            
            #     print('root_config.LOG_WHEN_SAMPLING==false...')
            #     devnull_file = open(os.devnull, 'w')
            #     original_stdout = sys.stdout
            #     sys.stdout = devnull_file
            x_samples_ddim = sample_model(input_im, models['turncam'], sampler, precision, h, w,
                                          ddim_steps, n_samples, scale, ddim_eta, used_x, y, z)
            # if(not root_config.LOG_WHEN_SAMPLING):
            #     sys.stdout = original_stdout
            #     devnull_file.close()
        output_ims = []
        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

        description = None

        if 'angles' in return_what:
            return (x, y, z, description, new_fig, show_in_im2, output_ims)
        else:
            return (description, new_fig, show_in_im2, output_ims)


def calc_cam_cone_pts_3d(polar_deg, azimuth_deg, radius_m, fov_deg):
    '''
    :param polar_deg (float).
    :param azimuth_deg (float).
    :param radius_m (float).
    :param fov_deg (float).
    :return (5, 3) array of float with (x, y, z).
    '''
    polar_rad = np.deg2rad(polar_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    fov_rad = np.deg2rad(fov_deg)
    polar_rad = -polar_rad  # NOTE: Inverse of how used_x relates to x.

    # Camera pose center:
    cam_x = radius_m * np.cos(azimuth_rad) * np.cos(polar_rad)
    cam_y = radius_m * np.sin(azimuth_rad) * np.cos(polar_rad)
    cam_z = radius_m * np.sin(polar_rad)

    # Obtain four corners of camera frustum, assuming it is looking at origin.
    # First, obtain camera extrinsics (rotation matrix only):
    camera_R = np.array([[np.cos(azimuth_rad) * np.cos(polar_rad),
                          -np.sin(azimuth_rad),
                          -np.cos(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(azimuth_rad) * np.cos(polar_rad),
                          np.cos(azimuth_rad),
                          -np.sin(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(polar_rad),
                          0.0,
                          np.cos(polar_rad)]])
    # print('camera_R:', lo(camera_R).v)

    # Multiply by corners in camera space to obtain go to space:
    corn1 = [-1.0, np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn2 = [-1.0, -np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn3 = [-1.0, -np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn4 = [-1.0, np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn1 = np.dot(camera_R, corn1)
    corn2 = np.dot(camera_R, corn2)
    corn3 = np.dot(camera_R, corn3)
    corn4 = np.dot(camera_R, corn4)

    # Now attach as offset to actual 3D camera position:
    corn1 = np.array(corn1) / np.linalg.norm(corn1, ord=2)
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    corn2 = np.array(corn2) / np.linalg.norm(corn2, ord=2)
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    corn3 = np.array(corn3) / np.linalg.norm(corn3, ord=2)
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1]
    corn_z3 = cam_z + corn3[2]
    corn4 = np.array(corn4) / np.linalg.norm(corn4, ord=2)
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4]

    return np.array([xs, ys, zs]).T


class ModelsGetter:
    models = None
    #param. B
    device=None
    ckpt=None
    config=None
    @classmethod
    def set_param(cls, device, ckpt, config):
        cls.device=device
        cls.ckpt=ckpt
        cls.config=config
        return None
    @classmethod
    def get_B(cls,  ):
        return cls.get( cls.device, cls.ckpt, cls.config)
    #
    @classmethod
    def to(cls,device):
        if  cls.models is None:
            return
        cls.models['turncam'].to(device)
        # cls.models['carvekit'].to(device)
    # @classmethod
    # def to_cpu(cls,):
    #     cls.to("cpu")
    # @classmethod
    # def to_gpu(cls,):
    #     cls.to(root_config.DEVICE)
    @classmethod
    def release_gmem(cls):
        cls.to("cpu")
        torch.cuda.empty_cache()
    @classmethod
    def get(cls, device, ckpt, config):
        if (cls.models == None):
            # print("[ModelGetter]cls.model==None, loading model...")
            config = OmegaConf.load(config)

            # Instantiate all models beforehand for efficiency.
            models = dict()
            print('Instantiating LatentDiffusion...')
            models['turncam'] = load_model_from_config(config, ckpt, device=device)#  see zero123/zero1/configs/sd-objaverse-finetune-c_concat-256.yaml
            if   NO_CARVEKIT:
                pass
            else:
                print('Instantiating Carvekit HiInterface...')
                models['carvekit'] = create_carvekit_interface()
            if CHECK_NSFW:
                print('Instantiating StableDiffusionSafetyChecker...')
                models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(
                    'CompVis/stable-diffusion-safety-checker').to(device)
                # Reduce NSFW false positives.
                # NOTE: At the time of writing, and for diffusers 0.12.1, the default parameters are:
                # models['nsfw'].concept_embeds_weights:
                # [0.1800, 0.1900, 0.2060, 0.2100, 0.1950, 0.1900, 0.1940, 0.1900, 0.1900, 0.2200, 0.1900,
                #  0.1900, 0.1950, 0.1984, 0.2100, 0.2140, 0.2000].
                # models['nsfw'].special_care_embeds_weights:
                # [0.1950, 0.2000, 0.2200].
                # We multiply all by some factor > 1 to make them less likely to be triggered.
                models['nsfw'].concept_embeds_weights *= 1.07
                models['nsfw'].special_care_embeds_weights *= 1.07
                print('Instantiating AutoFeatureExtractor...')
                models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
                    'CompVis/stable-diffusion-safety-checker')

            cls.models = models
        else:
            cls.to(device)
        return cls.models


from util_4_e2vg import ImagePathUtil
from util_4_e2vg.Util import get_xyzLinearGradient

def sample_model_(
        device_idx=_GPU_INDEX,
        config=CONFIG,
                                                ddim_steps=50,
        **kw):
    ckpt = root_config.weightPath_zero123
    # ckpt="/root/autodl-tmp/zero123-xl.ckpt"


    # from  import ImagePathUtil
    def predict_img(i, x, y, z, raw_im, path_output_ims, n_samples=4):
        if not os.path.exists(path_output_ims):
            os.makedirs(path_output_ims)
        # check exist
        for j in range(n_samples):
            img_save_path = os.path.join(path_output_ims, ImagePathUtil.get_path(i, j, x, y, z, ))
            if not os.path.exists(img_save_path) or root_config.FORCE_zero123_render_even_img_exist:
                pass
            else:
                return
        # render
        models = ModelsGetter.set_param(device, ckpt, config)
        x, y, z, description, new_fig, show_in_im2, output_ims = main_run(models, device, None, 'angles_gen',
                                                                          x, y, z, raw_im=raw_im,
                                                                          n_samples=n_samples)
        for j, output_im in enumerate(output_ims):
            img_save_path = os.path.join(path_output_ims, ImagePathUtil.get_path(i, j, x, y, z, ))
            output_im.save(img_save_path)

    def batch_predict_img(xyzLinearGradient, raw_im, path_output_ims):
        if not os.path.exists(path_output_ims):
            os.makedirs(path_output_ims)
        # check exist
        new_xyzLinearGradient=[]
        j = 0
        for i, (x, y, z) in enumerate(xyzLinearGradient):
            img_save_path = os.path.join(path_output_ims, ImagePathUtil.get_path(i, j, x, y, z, ))
            if not os.path.exists(img_save_path) or root_config.FORCE_zero123_render_even_img_exist:
                new_xyzLinearGradient.append((x, y, z))
                ttt345=glob.glob(os.path.join(path_output_ims, ImagePathUtil.i2glob(i)))
                assert ttt345==[],f"不应有 other img ({ttt345}) that has the same i:{i}"
        # print("[batch_predict_img] new_xyzLinearGradient=",new_xyzLinearGradient)
        if(len(new_xyzLinearGradient)==0):
            return
        else:
            assert len(new_xyzLinearGradient)==len(xyzLinearGradient),f"目前output_ims是全部gen完才保存，故要么new_xyzLinearGradient==[],要么new_xyzLinearGradient与xyzLinearGradient一样。new_xyzLinearGradient= {new_xyzLinearGradient}\nxyzLinearGradient= {xyzLinearGradient}"
        xyzLinearGradient=new_xyzLinearGradient
        # render
        models = ModelsGetter.set_param(device, ckpt, config)
        xs = []
        ys = []
        zs = []
        for i, (x, y, z) in enumerate(xyzLinearGradient):
            xs.append(x)
            ys.append(y)
            zs.append(z)
        _, _, _, _, _, _, output_ims = main_run(models, device, None, 'angles_gen',
                                                xs, ys, zs, raw_im=raw_im,
                                                n_samples=n_samples,
                                                ddim_steps=ddim_steps,
                                                batch_sample=True,
                                                )
        j = 0
        for i, output_im in enumerate(output_ims):
            x = xs[i]
            y = ys[i]
            z = zs[i]
            img_save_path = os.path.join(path_output_ims, ImagePathUtil.get_path(i, j, x, y, z, ))
            output_im.save(img_save_path)


    if ("id" in kw):
        id = kw["id"]
        # args: id, input_image_path, num_samples
        if 'parentFolderName_output_ims' in kw:
            parentFolderName_output_ims=kw['parentFolderName_output_ims']
            assert '/' not in parentFolderName_output_ims
            assert isinstance(parentFolderName_output_ims,str)
            assert len(parentFolderName_output_ims)>0
            assert parentFolderName_output_ims.startswith('_'),"建议startswith('_')以示区分"
            path_output_ims = os.path.join(root_config.dataPath_zero123, parentFolderName_output_ims,id)
        else:
            path_output_ims = os.path.join(root_config.dataPath_zero123, id)
        if not os.path.exists(path_output_ims):
            os.makedirs(path_output_ims)
        raw_im = Image.open(kw["input_image_path"])
        n_samples = kw["num_samples"]
        if ("l_xyz" in kw and kw["l_xyz"]):
            xyzLinearGradient = kw["l_xyz"]
        else:
            assert 0, "l_xyz呢朋友"
    else:
        OBJ_NAME = "cup"
        raw_im = Image.open(f'{OBJ_NAME}.jpg')  # PIL image  0-255   H,W,3,not np.array
        path_output_ims = f"output_{OBJ_NAME}"
        Z = 0
        xyzLinearGradient = get_xyzLinearGradient(
            xyzStops=(
                (0, -180, Z, 0.0),
                (0, 180, Z, 0.5),
                (-80, 180, Z, 1.0),
            ),
            N=128,
        )
        n_samples = 4

    path_output_ims = os.path.abspath(path_output_ims)
    if ("batch_sample" in kw and kw["batch_sample"]):
        batch_predict_img(xyzLinearGradient, raw_im=raw_im, path_output_ims=path_output_ims, )
    else:
        for i, (x, y, z) in enumerate(xyzLinearGradient):
            predict_img(i, x, y, z, raw_im=raw_im, path_output_ims=path_output_ims, n_samples=n_samples)
    ModelsGetter.release_gmem()
    return path_output_ims 


if __name__ == '__main__':
    e2vg_run_demo()
