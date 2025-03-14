python tools/dataset_patch_split.py \
--dataset-type "potsdam" \
--img-dir "data/potsdam/2_Ortho_RGB" \
--mask-dir "data/potsdam/5_Labels_all" \
--output-img-dir "data/potsdam/train/images_1024" \
--output-mask-dir "data/potsdam/train/masks_1024" \
--split-size 1024 \
--stride 512 \
--mode "train"

python tools/dataset_patch_split.py \
--dataset-type "potsdam" \
--img-dir "data/potsdam/2_Ortho_RGB" \
--mask-dir "data/potsdam/5_Labels_all_noBoundary" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024" \
--split-size 1024 \
--stride 1024 \
--mode "test"

python tools/dataset_patch_split.py \
--dataset-type "potsdam" \
--img-dir "data/potsdam/2_Ortho_RGB" \
--mask-dir "data/potsdam/5_Labels_all" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024_RGB" \
--split-size 1024 \
--stride 1024 \
--mode "test"