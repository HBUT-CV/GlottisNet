_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]

pretrained = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_small-8427ecf0.pth'

model = dict(
    type='DDOD',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='MobileNetV3', 
        arch='small',
        out_indices=(3, 8, 11), # Modify out_indices
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone.')), 
    neck=dict(
        type='FPN',
        in_channels=[24, 48, 96], #[16, 32, 96, 1280],
        out_channels=64,
        start_level=0, # 1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='DDODHead',
        num_classes=8,
        in_channels=64,
        stacked_convs=4,
        feat_channels=64,
        use_dcn=False,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_iou=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        # assigner is mean cls_assigner
        assigner=dict(type='ATSSAssigner', topk=9, alpha=0.8),
        reg_assigner=dict(type='ATSSAssigner', topk=9, alpha=0.5),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
