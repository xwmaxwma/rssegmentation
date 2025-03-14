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

######################## dataset_config ######################
exp_name = "work_dirs/docnet_potsdam"
_base_ = '../_base_/potsdam_config.py'
epoch = 80
num_class = 6
ignore_index = 6

######################### model_config #########################
model_config = dict(
    backbone = dict(
        type = 'get_hrnetv2_w32'
    ),
    seghead = dict(
        type = 'DOC_Head',
        num_class = num_class
    ),
    classifier = dict(
        type = 'Base_Classifier',
        transform_channel = 512,
        num_class = num_class,
    ),
    upsample=dict(
        type='Interpolate',
        mode='bilinear',
        scale=[4, 4],
    )
)
loss_config = dict(
    type = 'myLoss',
    loss_name = ['CELoss', 'CELoss'],
    loss_weight = [1, 0.4],
    ignore_index = ignore_index
)


######################## optimizer_config ######################
optimizer_config = dict(
    optimizer = dict(
        type = 'AdamW',
        backbone_lr = 0.0001,
        backbone_weight_decay = 0.05,
        lr =  0.0001,
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
