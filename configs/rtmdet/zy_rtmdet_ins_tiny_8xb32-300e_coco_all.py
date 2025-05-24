_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/coco_detection.py', './rtmdet_tta.py'
]

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        use_depthwise=True,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(
        type='ZY_CSPNeXtPAFPN',
        in_channels=[96, 192, 384], 
        out_channels=96,
        use_depthwise=True,  
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        # _delete_=True,
        type='RTMDetInsSepBNHead',
        num_classes=5, #1,
        in_channels=96,
        feat_channels=96,
        stacked_convs=0,
        share_conv=True,
        pred_kernel_size=1,
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='DiceLoss', loss_weight=2.0, eps=5e-6, reduction='mean'),
        with_objectness=False,
        # exp_on_reg=True,
        ),
    
    train_cfg=dict(
        assigner=dict(type='ZY_DynamicSoftLabelAssigner', 
                      topk=3,
                      soft_cls_weight=1, # 1,
                      iou_weight=3, # 3,
                      soft_center_weight=3, #1
                      ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5),
)


train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}), # type: ignore
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    # 会随机复制粘贴图像
    # dict(
    #     type='CachedMosaic',
    #     img_scale=(256, 256),
    #     pad_val=57.0,
    #     max_cached_images=20,
    #     random_pop=False),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    # dict(type='Resize', scale=(256, 256), keep_ratio=False),
    # dict(
    #     type='RandomResize',
    #     scale=(256, 256), # 320
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    
    # add
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18), 
    ###
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(57, 57, 57))),
    # dict(type='Pad', size=(128, 256), pad_val=dict(img=(57, 57, 57))),
    #! Rerun包含此项，貌似会导致mask前期就消失
    # dict(
    #     type='CachedMixUp',
    #     img_scale=(256, 256),
    #     ratio_range=(1.0, 1.0),
    #     max_cached_images=10,
    #     random_pop=False,
    #     pad_val=(57, 57, 57),
    #     prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

'''
train_pipeline_stage2 = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomResize',
        scale=(320, 320),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(320, 320),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(320, 320), pad_val=dict(img=(57, 57, 57))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(type='Pad', size=(320, 320), pad_val=dict(img=(57, 57, 57))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(type='RandomResize', scale=(256, 256), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
'''

train_pipeline_stage2 = train_pipeline

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}), # type: ignore
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    # dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(57, 57, 57))),
    # dict(type='Pad', size=(128, 256), pad_val=dict(img=(57, 57, 57))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=_base_.dataset_type, # type: ignore
            data_root=_base_.data_root, # type: ignore
            ann_file='annotations/instances_train2017.json',
            data_prefix=dict(img='train2017/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=16, num_workers=4, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

max_epochs = 500 #300
stage2_num_epochs = 50 #20
base_lr = 0.0005
interval = 1

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

val_evaluator = dict(proposal_nums=(100, 1, 10), metric=['bbox', 'segm'])
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(checkpoint=dict(
                    interval=1,
                    # max_keep_ckpts=-1,  
                    max_keep_ckpts=3, 
                    save_best='auto' ),
                    visualization=dict(type='DetVisualizationHook', draw=False))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]


# add for TensorBord
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')