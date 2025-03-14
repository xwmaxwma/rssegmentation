dataset = 'uavid'
dataset_config = dict(
    type='Uavid',
    data_root='data/uavid/tmp',
    train_mode=dict(
        transform=dict(
            RandomSizeAndCrop={"size": 512, "crop_nopad": False},
            RandomHorizontallyFlip=None,
            RandomVerticalFlip=None,
            RandomRotate={"degree": 0.2},
        ),
        loader=dict(
            batch_size=4,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        ),
    ),

    val_mode=dict(
        transform=dict(
            Resize={'size': 512}
        ),
        loader=dict(
            batch_size=4,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
    ),
)

metric_cfg1 = dict(
            task = 'multiclass',
            average='micro',
            num_classes = 9, 
            ignore_index = 8
        )

metric_cfg2 = dict(
            task = 'multiclass',
            average='none',
            num_classes = 9, 
            ignore_index = 8
        )
