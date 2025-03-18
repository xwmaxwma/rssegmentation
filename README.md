![cap](./cap.jpg)

# 🔥 News

- `2025/3/14`: We fix some bugs and add **CAM** and **Throughput**.
- `2025/2/11`: [LOGCAN++](https://arxiv.org/abs/2406.16502) has been accepted by TGRS2025!
- `2025/1/24`: [SCSM](https://arxiv.org/abs/2501.13130) has been accepted by ISPRS2025!
- `2024/10/11`: [SSA-Seg](https://arxiv.org/abs/2405.06525) has been accepted by NeurIPS2024! It is an effective and powerful classifier for semantic segmentation. We recommend interested researchers to optimize it for semantic segmentation in remote sensing, which is a promising direction.

# 📷 Introduction

**rssegmentation** is an open-source semantic segmentation toolbox, which is dedicated to reproducing and developing advanced methods for semantic segmentation of remote sensing images.

<div align="center">
  <b>Overview</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>Methods</b>
      </td>
      <td>
        <b>Datasets</b>
      </td>
      <td>
        <b>Tools</b>
      </td>
    </tr>
	<tr valign="top">
      <td>
        <ul>
          <li><a href="https://ieeexplore.ieee.org/abstract/document/10095835/">LoG-Can(ICASSP2023) </a></li>
          <li><a href="https://ieeexplore.ieee.org/abstract/document/10219583/">SACANet(ICME2023)</a></li>
       		<li><a href="https://ieeexplore.ieee.org/abstract/document/10381808/">DOCNet(GRSL2024)</a></li>
          <li><a href="https://ieeexplore.ieee.org/document/10884928/">LOGCAN++(TGRS2025)</a></li>
          <li><a href="https://www.sciencedirect.com/science/article/pii/S0924271625000255?via%3Dihub">SCSM(ISPRS2025)</a></li>
          <li>CenterSeg(Under review)</a></li>
        </ul>
      </td>
<td>
        <ul>
          <li><a href="https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx">Vaihingen </a></li>
          <li><a href="https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx">Potsdam</a></li>
       		<li><a href="https://codalab.lisn.upsaclay.fr/competitions/421">LoveDA</a></li>
          <li>iSAID(to do)</a></li>
        </ul>
      </td>
<td>
        <ul>
          <li>Training </a></li>
          <li>Testing</a></li>
       		<li>Params, FLOPs, Latency, Throughput </a></li>
			<li>Class activation map </a></li>
			<li>TSNE map </a></li>
        </ul>
      </td>
</table>
**Note:  The checkpoints will be updated gradually**



# 📒 Folder Structure

<details>
<summary>
Prepare the following folders to organize this repo:
</summary>

```
rssegmentation
├── rsseg (core code for datasets and models)
├── tools (some useful tools)
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
</details>


# 🔐 Preparation

- **Environment**

```shell
conda create -n rsseg python=3.9
conda activate rsseg
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

- **Data Preprocess**

```shell
# Modify img-dir and mask-dir if necessary
bash tools/vaihingen_preprocess.sh 
bash tools/potsdam_preprocess.sh 
```

# 📚 Use example

### 1️⃣ Training

```shell
python train.py -c configs/vaihingen/logcanplus.py
```

### 2️⃣ Testing

- **Vaihingen and Potsdam**

```shell
python test.py \
-c configs/vaihingen/logcanplus.py \
--ckpt work_dirs/logcanplus_vaihingen/epoch=45.ckpt \
```

- **LoveDA**

Note that since the loveda dataset needs to be evaluated online, we provide the corresponding test commands.

```shell
python online_test.py \
-c configs/loveda/logcanplus.py \
--ckpt work_dirs/logcanplus_loveda/epoch=45.ckpt \
```

### 3️⃣ Useful tools

- **Param and FLOPs**

```shell
python tools/flops_params_count.py -c configs/vaihingen/logcanplus.py 
```
- **Latency**

```shell
python tools/latency_count.py \
-c configs/vaihingen/logcanplus.py \
--ckpt work_dirs/logcanplus_vaihingen/epoch=45.ckpt \
```
- **Throughput**

```shell
  python tools/throughput_count.py -c configs/vaihingen/logcanplus.py
```

  

- **Class activation map**

```shell
python tools/cam.py \
-c configs/vaihingen/logcanplus.py \
--ckpt work_dirs/logcanplus_vaihingen/epoch=45.ckpt \
--tar_layer "model.net.seghead.catconv2[-2]" \
--tar_category 1
```

- **Tsne map**

```shell
python tools/tsne.py \
-c configs/vaihingen/logcanplus.py \
--ckpt work_dirs/logcanplus_vaihingen/epoch=45.ckpt 
```

# 🌟 Citation

<details>
<summary>
If you find our repo useful for your research, please consider giving a 🌟 and citing our work below.
</summary>

```
@inproceedings{logcan,
  title={Log-can: local-global class-aware network for semantic segmentation of remote sensing images},
  author={Ma, Xiaowen and Ma, Mengting and Hu, Chenlu and Song, Zhiyuan and Zhao, Ziyan and Feng, Tian and Zhang, Wei},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
@inproceedings{sacanet,
  title={Sacanet: scene-aware class attention network for semantic segmentation of remote sensing images},
  author={Ma, Xiaowen and Che, Rui and Hong, Tingfeng and Ma, Mengting and Zhao, Ziyan and Feng, Tian and Zhang, Wei},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={828--833},
  year={2023},
  organization={IEEE}
}

@article{docnet,
  title={DOCNet: Dual-Domain Optimized Class-Aware Network for Remote Sensing Image Segmentation},
  author={Ma, Xiaowen and Che, Rui and Wang, Xinyu and Ma, Mengting and Wu, Sensen and Feng, Tian and Zhang, Wei},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2024},
  publisher={IEEE}
}

@ARTICLE{logcan++,
  author={Ma, Xiaowen and Lian, Rongrong and Wu, Zhenkai and Guo, Hongbo and Yang, Fan and Ma, Mengting and Wu, Sensen and Du, Zhenhong and Zhang, Wei and Song, Siyang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={LOGCAN++: Adaptive Local-Global Class-Aware Network For Semantic Segmentation of Remote Sensing Images}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2025.3541871}}

@article{scsm,
title = {A novel scene coupling semantic mask network for remote sensing image segmentation},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {221},
pages = {44-63},
year = {2025},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2025.01.025},
url = {https://www.sciencedirect.com/science/article/pii/S0924271625000255},
author = {Xiaowen Ma and Rongrong Lian and Zhenkai Wu and Renxiang Guan and Tingfeng Hong and Mengjiao Zhao and Mengting Ma and Jiangtao Nie and Zhenhong Du and Siyang Song and Wei Zhang},
}
```
</details>

# 📮 Contact

If you are confused about the content of our paper or look forward to further academic exchanges and cooperation, please do not hesitate to contact us. The e-mail address is xwma@zju.edu.cn. We look forward to hearing from you!

# 💡 Acknowledgement

Thanks to previous open-sourced repo:

- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [pytorch lightning](https://lightning.ai/)
- [fvcore](https://github.com/facebookresearch/fvcore)
