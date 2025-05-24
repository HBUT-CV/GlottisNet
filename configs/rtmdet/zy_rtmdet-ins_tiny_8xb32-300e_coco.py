_base_ = './rtmdet-ins_l_8xb32-300e_coco.py'
# _base_ = ['./rtmdet-ins_s_8xb32-300e_coco.py', '../_base_/datasets/coco_instance.py']

# dataset_type = 'CocoDataset_ins'
dataset_type = 'CocoDataset'
# data_root = 'data/coco_ins_f/'
data_root = 'data/coco_ins/'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375,
        use_depthwise=True, # zy add
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(type='ZY_CSPNeXtPAFPN',
              in_channels=[96, 192, 384], 
                out_channels=96,
                use_depthwise=True,  
                num_csp_blocks=1),
    bbox_head=dict(num_classes=5,  
                    stacked_convs=0, 
                    in_channels=96,
                    feat_channels=96,
                    # use_depthwise=True,
                    ),
    train_cfg=dict(
        assigner=dict(type='ZY_DynamicSoftLabelAssigner', 
                      topk=3,
                      soft_cls_weight=1, # 1,
                      iou_weight=3, # 3,
                      soft_center_weight=3, #1
                      )))

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
        img_scale=(640, 640),
        pad_val=57.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(640, 640), # 320
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
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
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(57, 57, 57),
        prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=32, #128,
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

val_dataloader = dict(
    batch_size=16,#1, 
    num_workers=5,#2,
    dataset=dict(
            type=dataset_type,
            data_root=data_root))


test_dataloader = val_dataloader

max_epochs = 500 #300
stage2_num_epochs = 50 #20
base_lr = 0.0001
interval = 1

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])


default_hooks = dict(checkpoint=dict(
        interval=1,
        # max_keep_ckpts=-1,  
        max_keep_ckpts=3, 
        save_best='auto' ),
        visualization=dict(type='DetVisualizationHook', draw=False))

# log_config = dict( 
#     interval=1, 
#     hooks=[ 
#         dict(type='TextLoggerHook'), 
#         # dict(type='WandbLoggerHook', init_kwargs=dict(project='Your-project')), 
#         dict(type='NeptuneLoggerHook', init_kwargs=dict(project='y-zhou/RTMDet')),
#         dict(type='TensorboardLoggerHook')
#     ]) 

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
