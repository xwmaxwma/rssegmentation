
######################## base_config #########################
gpus = [1]
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
logging_interval = 'epoch'
resume_ckpt_path = None
pretrained_ckpt_path = None
monitor = 'val_miou'

test_ckpt_path = None

exp_name = "work_dirs/sacanet_hrnet_loveda_test"

######################## dataset_config ######################
ALL_DATASET = {
    'vaihingen': (150, 6, 6),
    'potsdam': (80, 6, 6),
    'loveda': (70, 7, -1),
}  

dataset = 'loveda'
_base_ = './_base_/loveda_config.py'
epoch, num_class, ignore_index = ALL_DATASET[dataset]

######################### model_config #########################
model_config = dict(
    backbone = dict(
        type = 'get_hrnetv2_w32',
    ),
    seghead = dict(
        type = 'Sacanet',
        num_class = num_class,
        patch_size=(16, 16),
    ),
    classifier = dict(
        type = 'Base_Classifier',
        transform_channel = 128,
        num_class = num_class,
    ),
    upsample=dict(
        type='Interpolate',
        mode='bilinear',
        scale=[4,4],
    )
)
loss_config = dict(
    type = 'myLoss',
    loss_name = ['CELoss', 'CELoss'],
    loss_weight = [1, 0.8],
    ignore_index = ignore_index
)

######################## optimizer_config ######################
optimizer_config = dict(
    optimizer = dict(
        type = 'SGD',
        backbone_lr = 0.001,
        backbone_weight_decay = 1e-4,
        lr = 0.01,
        weight_decay = 1e-4,
        momentum = 0.9,
        lr_mode = "single"
    ),
    scheduler = dict(
        type = 'Poly',
        poly_exp = 0.9,
        max_epoch = epoch
    )
)
