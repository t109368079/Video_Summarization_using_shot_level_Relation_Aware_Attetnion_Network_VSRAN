# Video Summarization using Shot-level Relation-Aware Attention Network (VSRAN)

## Introduction

![](https://i.imgur.com/q3INdCQ.png)
Video Summarization is a task that extract the most important part of a video. Previous researchers focus on frame-level video summarization which fail to consider the action or transition amoung frames. 
In this research, we propose a pipeline for shot-based video summarization. Shot are represent with 2D (static) and 3D (motion) CNN. Relation-Aware Attention are used to fused motion features with static features. A regression model predict shot importance score based on fused shot feature. Two key-shot selection methods are used to maximize the summary shot score with limit length.
For full paper, please click [here](https://drive.google.com/file/d/1x19kPrfBahyvyxVpSuSxNnSgmF-JUbyw/view?usp=share_link).

## Extracted Feature h5 files Download Link
Please download the pre-extracted feature h5 files using this [link](https://drive.google.com/drive/folders/1Czq5oTXvFiz6SKFdACLetxjds84nlHjY?usp=share_link).

### Directory tree
```
VSRAN
├──README.md
├──datasets (put h5 file here-recommend)
├──split_folder
├──models
| ├──layer.py
| ├──model.py
| ├──relation_aware_attention.py
├──utils
| ├──config.py
| ├──CoSum+DataLoader.py
| ├──f1_score_metrics.py
| ├──generate_cosum_split.py
| ├──generate_greedy_summary.py
| ├──kfold_split.py
| ├──knapsack.py
| ├──loss_metrics.py
| ├──metrics.py
| ├──My_Dataset.py
| ├──video_summarization.py
| ├──video_summarization_dataset.py
├──main.py
├──transfer_main.py
├──Augment_main.py
├──trainer.py
├──config.yaml
├──transfer_config.yaml
├──augment_config.yaml
```



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

### Score Prediction based evaluation
1. Kendall coefficient   
2. Spearman coefficient   
3. mAP
 
#### SumMe Dataset
| Method \ Metrics |  Kendall  | Spearman  |
|:---------------:|:---------:|:---------:|
|     Random      |    0.0    |    0.0    |
|     DR-DSN      |   0.047   |   0.048   |
|      RSGN       |   0.083   |   0.085   |
| Proposed VSRAN  | **0.104** | **0.123** |

#### CoSum Dataset
| Method \ Metrics | mAP-top5  | mAP-top15 |
|:---------------:|:---------:|:---------:|
|       KTS       |   0.684   |   0.686   |
|     seqDPP      |   0.692   |   0.709   |
|     SubMod      |   0.735   | **0.745** |
|    DeSumNet     |   0.721   |   0.736   |
| Proposed VSRAN  | **0.792** |   0.676   |

### Summary based evaluation
#### SumMe Dataset
|  Method \ Setting  | Standard | Transfer | Augment  |
|:------------------:|:--------:|:--------:|:--------:|
| Random Frame level  <td spancol=3> 41.0
| Random Shot level  |   34.7   |          |          |
|      SUM-GAN       |   41.7   |   43.6   |          |
|       VASNet       |   43.4   |   42.5   |   41.9   |
|        RSGN        |   45.0   |   45.7   |   44.0   |
|      Clip-it       |   52.5   | **54.7** |   50.0   |
|   Proposed VSRAN   | **57.7** |   45.5   | **54.7** |
|    ground truth    |   64.7   |          |          |


