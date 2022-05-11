# Grouped SSD (GSSD) and GSSD++ for liver lesion detection from multi-phase CT

This repository is an official PyTorch implementation of the papers:

> Sang-gil Lee, Jae Seok Bae, Hyunjae Kim, Jung Hoon Kim, Sungroh Yoon. "Liver Lesion Detection from Weakly-labeled Multi-phase CT Volumes with a Grouped Single Shot MultiBox Detector." _MICCAI_ (2018).
>[[arxiv]](https://arxiv.org/abs/1807.00436)
>

> Sang-gil Lee, Eunji Kim, Jae Seok Bae, Jung Hoon Kim, Sungroh Yoon. "Robust End-to-End Focal Liver Lesion Detection using Unregistered Multiphase Computed Tomography Images." _IEEE TETCI_ (2021).
>[[arxiv]](https://arxiv.org/abs/2112.01535)
>
>

![](./pics/gssdpp5.png)
![](./pics/gssdpp1.png)

## Overview
This project provides an application of high-speed single-stage object detection models (e.g., SSD) to a computer-aided diagnosis (CAD) of focal liver lesions.

The use of single-stage detection model enables real-time plug-in support to the CAD toolkit in an interactive manner. With the design principles inspired by the real-world clinical practice of CAD performed by radiologists, this project helps radiologists to focus on the machine-predicted region of interest and can reduce human error to the small and hard-to-see lesions.

GSSD and GSSD++ present custom-designed neural submodules (grouped convolutions, 1x1 channel fusion, and an attention-guided multiphase alignment of the unregistered CT images) to maximize performance designed for the medical object detection from the multi-phase CT images.


## Data

This repo uses multi-phase CT dataset collected from a university hospital, which is unfortunately outside of my authority to release it to the public.

During the custom data preparation, we followed the data structure of the liver segmentation label nearly identical to 3Dircadb (http://www.ircad.fr/research/3dircadb/).

The input data of the lesion detection and the corresponding label look like the figures from our [MICCAI 2018 paper](https://arxiv.org/abs/1807.00436).
![](./pics/figure1.png)
![](./pics/gssdpp6.png)


## Model and Training 

`ssd_liverdet/models/ssd_multiphase_custom_group.py`: Implementation of our main model, GSSD++.

Based on the popular single-stage objection detection model, SSD, it applies a phase-wise grouped convolution followed by the 1x1 channel fusion module. The model also predicts spatial offsets of the registration mismatch in feature space, guided by visual cues from the phase-wise self-attention.

`ssd_liverdet/pixel_link/model.py`: Our implementation of the neural submodules applied to PixelLink, an alternative backbone CNN architecture which showed improved lesion detection performance from the previous work.

`ssd_liverdet/train_lesion_multiphase_v2.py`: GSSD or GSSD++ training for liver lesion detection. Assumes 4-phase 512x512 slices of dicom CT image and their corresponding bounding box annotations.

`ssd_liverdet/train_lesion_multiphase_v2_pixellink.py`: PixelLink++ training for liver lesion detection. Assumes 4-phase 512x512 slices of dicom CT image and their corresponding bounding box annotations.


We used the following command that includes hyperparameters used in GSSD and GSSD++.

```bash
# The following command trains GSSD, a multi-phase focal lesion detection model used in the MICCAI 2018 paper.
python train_lesion_multiphase_v2.py \
--datapath (path_to_dataset) \
--output (path_to_checkpoint_and_log) \
--load_data_to_ram False \
--ssd_type gssd \
--use_normalize True \
--use_self_attention False --use_self_attention_base False \
--num_dcn_layers 0 --dcn_cat_sab False \
--batch_size 32 \
--max_iter 60000 --stepvalues 30000 50000 --gamma 0.1 --lr 0.001 \
--gt_pixel_jitter 0.0 --num_workers 2 --speedrun 4999 \
--ohnm_neg_ratio 3 \
--cross_validation 1 \
--use_07_metric False --confidence_threshold 0.2 \
--ap_list "0.3,0.5,0.7" --iobb_list "0.3,0.5,0.7" \
--model_name GSSD
```

```bash
# The following command trains GSSD++, an improved GSSD used in the IEEE TETCI 2021 paper.
python train_lesion_multiphase_v2.py \
--datapath (path_to_dataset) \
--output (path_to_checkpoint_and_log) \
--load_data_to_ram False \
--ssd_type gssd \
--use_normalize True \
--use_self_attention True --use_self_attention_base True \
--groups_dcn 4 --num_dcn_layers 1 --dcn_cat_sab True \
--modify_dcn_lr True \
--batch_size 32 \
--max_iter 60000 --stepvalues 30000 50000 --gamma 0.1 --lr 0.001 \
--gt_pixel_jitter 0.0 --num_workers 2 --speedrun 4999 \
--ohnm_neg_ratio 3 \
--cross_validation 1 \
--use_07_metric False --confidence_threshold 0.2 \
--ap_list "0.3,0.5,0.7" --iobb_list "0.3,0.5,0.7" \
--model_name GSSD_plusplus
```

## Performance

GSSD++ and PixelLink++ exhibit state-of-the-art detection results measured by IoU and IoBB under various overlap thresholds. Our models also provide robustness to registration mismatch of the multi-phase data using the attention-guided multi-phase alignment module.

![](./pics/gssdpp4.png)

## Visualization

The following figures visualize the detection output and the attention map from GSSD++. 
![](./pics/gssdpp3.png)
![](./pics/gssdpp2.png)


## Reference
The original SSD implementation is hard-forked from [here](https://github.com/amdegroot/ssd.pytorch).

PixelLink implementation is hard-forked from [here](https://github.com/cheerss/PixelLink-with-pytorch).
