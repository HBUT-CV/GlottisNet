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
    bbox_head=dict(num_classes=1, #5, 
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
    dict(type='LoadImageFromFile', file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]


val_dataloader = dict(
    batch_size=1, 
    num_workers=2,
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
        save_best='auto' 
    ))

# log_config = dict( 
#     interval=1, 
#     hooks=[ 
#         dict(type='TextLoggerHook'), 
#         # dict(type='WandbLoggerHook', init_kwargs=dict(project='Your-project')), 
#         dict(type='NeptuneLoggerHook', init_kwargs=dict(project='y-zhou/RTMDet')),
#         dict(type='TensorboardLoggerHook')
#     ]) 

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend', save_dir="/home/ren2/data3/ZhangYang/mmdetection_v3x_glottis/work_dirs_ins/zy_run/320/zy_rtmdet-ins_tiny_detcov0_MyBlock1_OnlyNeck_k=5_FLassign133Topk3")]
