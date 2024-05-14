
# Introduction

**rssegmentation** is an open-source semantic segmentation toolbox, which is dedicated to reproducing and developing advanced methods for semantic segmentation of remote sensing images.

- Supported Methods
  - [LOGCAN](https://ieeexplore.ieee.org/abstract/document/10095835/) (ICASSP2023)

  - [SACANet](https://ieeexplore.ieee.org/abstract/document/10219583/) (ICME2023)

  - [DOCNet](https://ieeexplore.ieee.org/abstract/document/10381808) (GRSL2024)

  - CenterSeg (Under review)

  - LOGCAN++ (Under review)

  - SCSMNet (Under review)
- Supported Datasets
  - [Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)
  - [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)
  - [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)
  - iSAID (update soon)
- Supported Tools
  - Training
  - Testing
  - Params and FLOPs counting
  - Class activation maps (Updated soon)

# Get start

```shell
conda create -n rsseg python=3.9
conda activate rsseg
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

# Folder Structure

Prepare the following folders to organize this repo:

```none
rssegmentation
├── rssegmentation (code)
├── work_dirs (save the model weights and training logs)
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original labels)
│   │   │   ├── Rural
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original labels)
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   ├── vaihingen
│   │   ├── ISPRS_semantic_labeling_Vaihingen 
│   │   │   ├── top (original images)
│   │   ├── ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE (original labels)
│   │   ├── ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE (original noBoundary lables)
│   │   ├── train (processed)
│   │   ├── test (processed)
│   ├── potsdam (the same with vaihingen)
│   │   ├── 2_Ortho_RGB (original images)
│   │   ├── 5_Labels_all (original labels)
│   │   ├── 5_Labels_all_noBoundary (original noBoundary lables)
│   │   ├── train (processed)
│   │   ├── test (processed)
```

# Data Processing

## Vaihingen

**train**

```shell
python tools/dataset_patch_split.py \
--dataset-type "vaihingen" \
--img-dir "/home/xwma/data/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top" \
--mask-dir "/home/xwma/data/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE" \
--output-img-dir "data/vaihingen/train/images_1024" \
--output-mask-dir "data/vaihingen/train/masks_1024" \
--split-size 1024 \
--stride 512 \
--mode "train"
```

**test and val**

```shell
python tools/dataset_patch_split.py \
--dataset-type "vaihingen" \
--img-dir "/home/xwma/data/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top" \
--mask-dir "/home/xwma/data/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024_RGB" \
--split-size 1024 \
--stride 1024 \
--mode "test"
```

```shell
python tools/dataset_patch_split.py \
--dataset-type "vaihingen" \
--img-dir "/home/xwma/data/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top" \
--mask-dir "/home/xwma/data/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024" \
--split-size 1024 \
--stride 1024 \
--mode "test"
```

## potsdam

**train**

```shell
python tools/dataset_patch_split.py \
--dataset-type "potsdam" \
--img-dir "/home/xwma/data/Potsdam/2_Ortho_RGB" \
--mask-dir "/home/xwma/data/Potsdam/5_Labels_all" \
--output-img-dir "data/potsdam/train/images_1024" \
--output-mask-dir "data/potsdam/train/masks_1024" \
--split-size 1024 \
--stride 512 \
--mode "train"
```

**test and val**

```shell
python tools/dataset_patch_split.py \
--dataset-type "potsdam" \
--img-dir "/home/xwma/data/Potsdam/2_Ortho_RGB" \
--mask-dir "/home/xwma/data/Potsdam/5_Labels_all_noBoundary" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024" \
--split-size 1024 \
--stride 1024 \
--mode "test"
```

```shell
python tools/dataset_patch_split.py \
--dataset-type "potsdam" \
--img-dir "/home/xwma/data/Potsdam/2_Ortho_RGB" \
--mask-dir "/home/xwma/data/Potsdam/5_Labels_all" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024_RGB" \
--split-size 1024 \
--stride 1024 \
--mode "test"
```


# Training

```shell
python train.py -c "configs/logcan.py"
```

# Testing

**Vaihingen and Potsdam**

```shell
python test.py \
-c "configs/logcan.py" \
--ckpt "work_dirs/LoGCAN_ResNet50_Loveda/epoch=45.ckpt" \
```

**LoveDA**
Note that since the loveda dataset needs to be evaluated online, we provide the corresponding test commands.
```shell
python online_test.py \
-c "configs/logcan.py" \
--ckpt "work_dirs/LoGCAN_ResNet50_Loveda/epoch=45.ckpt" \
```

# Useful tools

We provide two useful commands to test the model for parameters, flops and latency.
```shell
python tools/flops_params_count.py \
-c "configs/logcan.py" \
```
```shell
python tools/latency_count.py \
-c "configs/logcan.py" \
--ckpt "work_dirs/LoGCAN_ResNet50_Loveda/epoch=45.ckpt" \
```
We will support feature visualizations as well as attention relationship visualizations soon.

# Citation

If you find our repo useful for your research, please consider citing our paper:

```
@inproceedings{ma2023log,
  title={Log-can: local-global class-aware network for semantic segmentation of remote sensing images},
  author={Ma, Xiaowen and Ma, Mengting and Hu, Chenlu and Song, Zhiyuan and Zhao, Ziyan and Feng, Tian and Zhang, Wei},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}

@inproceedings{ma2023sacanet,
  title={Sacanet: scene-aware class attention network for semantic segmentation of remote sensing images},
  author={Ma, Xiaowen and Che, Rui and Hong, Tingfeng and Ma, Mengting and Zhao, Ziyan and Feng, Tian and Zhang, Wei},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={828--833},
  year={2023},
  organization={IEEE}
}

@article{ma2024docnet,
  title={DOCNet: Dual-Domain Optimized Class-Aware Network for Remote Sensing Image Segmentation},
  author={Ma, Xiaowen and Che, Rui and Wang, Xinyu and Ma, Mengting and Wu, Sensen and Feng, Tian and Zhang, Wei},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2024},
  publisher={IEEE}
}
```

# Acknowledgement

Thanks to previous open-sourced repo:

- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [pytorch lightning](https://lightning.ai/)
- [fvcore](https://github.com/facebookresearch/fvcore)
