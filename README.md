## **Introduction**
Source code for **A Real-time Scale-robust Network for Glottis Segmentation in Nasal Transnasal Intubation**. For more details, please refer to our paper and [dataset](https://figshare.com/articles/journal_contribution/UAAL_Dataset_Upper_Airway_Anatomical_Landmark_Dataset_for_Automated_Bronchoscopy_and_Intubation/26342779/3) .

The source code is based on [MMDetection](https://github.com/open-mmlab/mmdetection).


## Installation

### Requirements

- Linux, Windows or macOS with Python ≥ 3.7, CUDA ≥ 10.1
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this


First install mmdet following the official guide: [INSTALL.md](https://mmdetection.readthedocs.io/en/latest/get_started.html).


Then build LosNet with:

```
cd YOUR_DIR/mmdetection_v3x_clinical
pip install -v -e .

# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

### Train Your Own Models

First, you need to be familiar with the config file. Please refer to [CONFIG](https://mmdetection.readthedocs.io/en/latest/user_guides/config.html).

We provide tools/train.py to launch training jobs on a single GPU. The basic usage is as follows.

```
python tools/train.py \
    ${CONFIG_FILE} \
    [optional arguments]
```

The process of training on the CPU is consistent with single GPU training. We just need to disable GPUs before the training process.

```
export CUDA_VISIBLE_DEVICES=-1

```

We provide tools/dist_train.sh to launch training on multiple GPUs. The basic usage is as follows.

```
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```

To test the trained model, you can simply run:

```
python tools/test.py ${CONFIG_FILE} work_dirs/PTH_FILE
```


