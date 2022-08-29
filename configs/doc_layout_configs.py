# the new config inherits the base configs to highlight the necessary modification
_base_ = './cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('Illustration', 'Text', 'ScienceText')
data_root_real = '/kaggle/input/real-doc/'
data_root_syn = '/kaggle/input/real-doc/'
img_norm_cfg = dict(
    mean=[248.78, 248.66, 248.60], std=[30.47, 30.78, 30.91], to_rgb=True)

albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='ImageCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap={
            'img': 'image',
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
syn_data = [
            (f"/kaggle/input/syn-v290-anns/anns_part/ann_part_{i}_processed_rename.json", f"/kaggle/input/syn-v290/part_{i}")
            for i in [30, 32, 33, 34]
            ]

dataset_syn = [dict(
            type=dataset_type,
            classes=classes,
            ann_file=ann_path,
            img_prefix=img_path,
            pipeline=train_pipeline)
            for ann_path, img_path in syn_data]

dataset_real = dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root_real + "real_anns/real_anns.json",
            img_prefix=data_root_real + 'images/images',
            pipeline=train_pipeline)
data_train = []
data_train.append(dataset_real)
data_train.extend(dataset_syn)
data_test = dict(
        type=dataset_type,
        classes=classes,
        ann_file=f"/kaggle/input/syn-v290-anns/v2.8.0_testing2_rename.json",
        img_prefix="/kaggle/input/syn-v290-anns/part_0",
        pipeline=test_pipeline),
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=data_train,
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=f"/kaggle/input/syn-v290-anns/v2.8.0_testing2_rename.json",
        img_prefix="/kaggle/input/syn-v290-anns/part_0",
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=f"/kaggle/input/syn-v290-anns/v2.8.0_testing2_rename.json",
        img_prefix="/kaggle/input/syn-v290-anns/part_0",
        pipeline=test_pipeline),
    )
evaluation = dict(metric=['bbox', 'segm'])

# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 3.
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 3.
                num_classes=3),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 3.
                num_classes=3),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 3.
                num_classes=3)],
    # explicitly over-write all the `num_classes` field from default 80 to 3.
    mask_head=dict(num_classes=3)))

# 3. training schedule settings
# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7])
# the max_epochs and step in lr_config need specifically tuned for the customized dataset
runner = dict(max_epochs=40)
log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'),
    dict(type='MMDetWandbHook',
         init_kwargs={'project': 'mmdetection'},
         interval=10,
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=100,
         bbox_score_thr=0.3),
])


workflow = [('train', 5)]