# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 15:51:32 2022


@author: Yuuki Misaki
"""

import os
import json
import h5py
import random
import numpy as np


def generate_split(video_list, n_split):
    total_video = len(video_list)
    validate_len = int(total_video/n_split) 
    video_list = [i for i in range(1, total_video+1)]
    random.shuffle(video_list)
    
    split = []
    for split_index in range(n_split):
        start = split_index*validate_len
        end = start + validate_len+1
        split.append({'train':video_list[:start]+video_list[end:], 'validate':video_list[start:end]})
    return split
    
def write_split(splits, path):
    split_file = open(path, 'w', encoding='utf-8')
    json.dump(splits, split_file)

dataset_path = '../datasets/cosum_goo3DRes_shot_center.h5'
dataset = h5py.File(dataset_path,'r')

video_list = list(dataset.keys())
split = generate_split(video_list, 4)
write_split(split, '../split_folder/coSum_0710_split.json')



with open('../datasets/Index_Name_Table_CoSum.txt', 'w') as f:
    for i, video_name in enumerate(video_list):
        f.write(f'{i}\t{video_name}\n')
        




