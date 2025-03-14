dataset = 'loveda'

dataset_config = dict(
    type='LoveDA',
    data_root='data/2021LoveDA',
    train_mode=dict(
        transform=dict(
            RandomSizeAndCrop = {"size": 512, "crop_nopad": False},
            RandomHorizontallyFlip = None,
            RandomVerticalFlip = None,
            RandomRotate = {"degree": 0.2},
            RandomGaussianBlur = None
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
            #Resize={'size': 512}
        ),
        loader=dict(
            batch_size=4,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
    ),
    test_mode=dict(
        transform=dict(
            # Resize={'size': 512}
        ),
        loader=dict(
            batch_size=4,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
    )
)

metric_cfg1 = dict(
            task = 'multiclass',
            average='micro',
            num_classes = 8,
            ignore_index = 7
        )

metric_cfg2 = dict(
            task = 'multiclass',
            average='none',
            num_classes = 8,
            ignore_index = 7
        )

eval_label_id_left = 0
eval_label_id_right = 6

class_name = ['building', 'road', 'water', 'barren', 'forest', 'agricultural', 'background']