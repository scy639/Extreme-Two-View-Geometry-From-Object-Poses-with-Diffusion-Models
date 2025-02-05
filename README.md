# Extreme-Two-View-Geometry-From-Object-Poses-with-Diffusion-Models

## [Paper](https://arxiv.org/abs/2402.02800)

## Introduction

Estimate relative camera pose of two images containing a co-visible object
![teaser](media/teaser.jpg)

## Setup
python >=3.8
1. run:
```
git clone https://github.com/scy639/Extreme-Two-View-Geometry-From-Object-Poses-with-Diffusion-Models.git
cd Extreme-Two-View-Geometry-From-Object-Poses-with-Diffusion-Models
pip install -r requirements.txt
mkdir -p install
cd install
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
cd ..
mkdir -p weight
cd weight
mkdir weight_gen6d
wget https://cv.cs.columbia.edu/zero123/assets/105000.ckpt
wget https://huggingface.co/One-2-3-45/code/resolve/main/one2345_elev_est/tools/weights/indoor_ds_new.ckpt
cd ..
```
if you have trouble installing Pytorch3D in the above way, follow https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md to install Pytorch3D

2. download gen6d weight
      - follow https://github.com/liuyuan-pal/Gen6D#Download to download gen6d_pretrain.tar.gz
      - tar -xvf  gen6d_pretrain.tar.gz
      - now you should have a folder called 'data', move sub folders 'detector_pretrain', 'selector_pretrain' and 'refiner_pretrain' to weight/weight_gen6d/



## Usage
### Evaluate on testset
Modify evaluate_on_testset.py and run it.
Before evaluating on GSO, you need to:
1. download gso-renderings.zip from https://drive.google.com/file/d/1fsMGFC3FdRFzWqClOT1jgbNRNqsE5sRv/view?usp=sharing and run 'unzip gso-renderings.zip'
2. configure src/path_configuration.py:
```
# the parent folder of GSO objects folders (GSO_alarm,GSO_backpack,...)
dataPath_gso='path/to/gso-renderings'
```
Before evaluating on NAVI, you need to:
1. follow https://github.com/google/navi/tree/49661e33598c4812584ef428a7b2019dbb318a3c to download navi_v1.tar.gz and extract 
2. configure src/path_configuration.py:
```
# the parent folder of NAVI objects folders (3d_dollhouse_sink,bottle_vitamin_d_tablets,...)
dataPath_navi=''
```
### Estimation on custom images
Modify infer_custom.py and run it.

<!-- ### Advance
For more config, refer to src/root_config.py -->

## Todo List
- [ ] Check setup
- [x] Upload GSO testset to a cloud drive
- [ ] Remove unused code; better document and comment
- [x] Remove unused package from requirement.txt
- [ ] Provide command line interface
- [ ] ...

## Acknowledgements
In this repository, we have used codes from the following repositories. We thank all the authors for sharing great codes.
- [Gen6D](https://github.com/liuyuan-pal/Gen6D)
- [zero123](https://github.com/cvlab-columbia/zero123)
- [One-2-3-45](https://github.com/One-2-3-45/One-2-3-45)
- [extrinsic2pyramid](https://github.com/demul/extrinsic2pyramid)
- [LoFTR](https://github.com/zju3dv/LoFTR)

## Citation
```
@misc{sun2024extreme,
      title={Extreme Two-View Geometry From Object Poses with Diffusion Models}, 
      author={Yujing Sun and Caiyi Sun and Yuan Liu and Yuexin Ma and Siu Ming Yiu},
      year={2024},
      eprint={2402.02800},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Follow-up work
[Generalizable Single-view Object Pose Estimation via Two-side Generation and Matching, WACV 2025 Oral](https://arxiv.org/abs/2411.15860) (improved accuracy but slower)

