
# Supported data
| Dataset | Original dataset | Preprocessed data | Description |
| ------- | ------------- | ----------------- | ----------- |
| Vaihingen | [link](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) | |
| Potsdam | [link](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) | |
| LoveDA | [link](https://codalab.lisn.upsaclay.fr/competitions/421) |

We provide links to the preprocessed data. Alternatively, you can preprocess the original dataset based on the following command.
We will support the isaid dataset soon.

# Supported Method
| Method | Config | Checkpoint |
| ------ | ------ | ---------- |
| LogCAN |
| SACANet |
| DocNet |

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
