dataset = 'vaihingen'
dataset_config = dict(
    type = 'Vaihingen',
    data_root = 'data/vaihingen',
    train_mode = dict(
        transform = dict(
            # RandomSizeAndCrop = {"size": 512, "crop_nopad": False},
            RandomScale = {'scale_list':[0.5, 0.75, 1.0, 1.25, 1.5], 'mode':'value'},
            SmartCropV1 = {'crop_size':512, 'max_ratio':0.75,
                                    'ignore_index':6, 'nopad':False},
            RandomHorizontallyFlip = None,
            RandomVerticalFlip = None,
            RandomRotate = {"degree": 0.2},
            RandomGaussianBlur = None
        ),
        loader = dict(
            batch_size = 4,
            num_workers = 4,
            pin_memory=True,
            shuffle = True,
            drop_last = True
        ),
    ),
    
    val_mode = dict(
        transform = dict(),
        loader = dict(
            batch_size = 4,
            num_workers = 4,
            pin_memory=True,
            shuffle = False,
            drop_last = False
        )
    ),
)
metric_cfg1 = dict(
            task = 'multiclass',
            average='micro',
            num_classes = 7, 
            ignore_index = 6
        )

metric_cfg2 = dict(
            task = 'multiclass',
            average='none',
            num_classes = 7, 
            ignore_index = 6
        )
class_name = ['ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter', 'Boundary']
eval_label_id_left = 0
eval_label_id_right = 5
