_base_ = './rtmdet-ins_s_8xb32-300e_coco.py'

dataset_type = 'CocoDataset_ins'
data_root = 'data/coco_ins/'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375,
        use_depthwise=True, # zy add
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(in_channels=[96, 192, 384], 
                out_channels=96,
                use_depthwise=True, # 使用DW 
                num_csp_blocks=1),
    bbox_head=dict(num_classes=5, 
                    stacked_convs=0, #头部堆叠卷积
                    in_channels=96,
                    feat_channels=96
                    ))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='CachedMosaic',
        img_scale=(320, 320),
        pad_val=57.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(400, 400),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(320, 320)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(320, 320), pad_val=dict(img=(57, 57, 57))),
    dict(
        type='CachedMixUp',
        img_scale=(320, 320),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(57, 57, 57),
        prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=128,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/instances_train2017.json',
            data_prefix=dict(img='train2017/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline)))

# zy add
val_dataloader = dict(
    batch_size=1, 
    num_workers=2,
    # batch_size=16, # 可能会导致精度有小范围波动
    # num_workers=8,
    dataset=dict(
            type=dataset_type,
            data_root=data_root)) # 注意修改dataset的type


# val_evaluator = dict(metric=['bbox', 'segm'],
#                      format_only=True,  # 只将模型输出转换为 coco 的 JSON 格式并保存
#                     outfile_prefix='./work_dirs/coco_detection/test')  # 要保存的 JSON 文件的前缀

test_dataloader = val_dataloader

max_epochs = 500 #300
stage2_num_epochs = 50 #20
base_lr = 0.002
interval = 1

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

default_hooks = dict(checkpoint=dict(
        interval=1,
        max_keep_ckpts=-1,  # 保存每一步模型
        # max_keep_ckpts=3,  # 保存3个模型
        save_best='auto' # 自动保存最佳模型
    ))
