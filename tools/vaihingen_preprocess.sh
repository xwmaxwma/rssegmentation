python tools/dataset_patch_split.py \
--dataset-type "vaihingen" \
--img-dir "data/vaihingen/ISPRS_semantic_labeling_Vaihingen/top" \
--mask-dir "data/vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE" \
--output-img-dir "data/vaihingen/train/images_1024" \
--output-mask-dir "data/vaihingen/train/masks_1024" \
--split-size 1024 \
--stride 512 \
--mode "train"

python tools/dataset_patch_split.py \
--dataset-type "vaihingen" \
--img-dir "data/vaihingen/ISPRS_semantic_labeling_Vaihingen/top" \
--mask-dir "data/vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024" \
--split-size 1024 \
--stride 1024 \
--mode "test"

python tools/dataset_patch_split.py \
--dataset-type "vaihingen" \
--img-dir "data/vaihingen/ISPRS_semantic_labeling_Vaihingen/top" \
--mask-dir "data/vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024_RGB" \
--split-size 1024 \
--stride 1024 \
--mode "test"