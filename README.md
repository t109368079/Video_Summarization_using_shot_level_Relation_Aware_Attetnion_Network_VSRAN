# Video Summarization using Shot-level Relation-Aware Attention Network (VSRAN)

## Introduction

![](https://i.imgur.com/q3INdCQ.png)
Video Summarization is a task that extract the most important part of a video. Previous researchers focus on frame-level video summarization which fail to consider the action or transition amoung frames. 
In this research, we propose a pipeline for shot-based video summarization. Shot are represent with 2D (static) and 3D (motion) CNN. Relation-Aware Attention are used to fused motion features with static features. A regression model predict shot importance score based on fused shot feature. Two key-shot selection methods are used to maximize the summary shot score with limit length.
For full paper, please click [here](https://drive.google.com/file/d/1x19kPrfBahyvyxVpSuSxNnSgmF-JUbyw/view?usp=share_link).

## Extracted Feature h5 files Download Link
Please download the pre-extracted feature h5 files using this [link]().

### Directory tree


## How to train in Standard Setting
1. In "config.yaml" file, change (1) "dataset_path", (2) "split_file_path"
2. Run python main.py 
3. The result of all evaluation metrics will be save to "report_path" indicate in "config.yaml"

## How to train in Transfer Setting
1. In "transfer_config.yaml" file, 
(1) Change/add augment h5 file(e.g. OVP or YouTube) path to "augment_dataset_path". 
(2) Add train h5 file path to(tvsum/summe/cosum) "train_dataset_path". 
(3) Add test h5 file to "test_dataset_path".
3. Run python transfer_main.py

## How to train in Augment Setting
1. In "augment_config.yaml" file,
(1) Change/add augment h5 file(any file that want to augment model) to "augment_dataset_path"
(2) Change "test_dataset_path"
(3) Change "split_file_path" to corresponse split file with test dataset
2. Run python Augment_main.py

## Result


