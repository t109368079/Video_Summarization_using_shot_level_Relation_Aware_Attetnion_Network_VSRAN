# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:26:58 2022

@author: Yuuki Misaki 
"""

import os
import json
import h5py
import numpy as np

import sys
sys.path.append('../')

from utils import video_summarization as tools

if __name__ == '__main__':
    dataset_path = '../datasets/tvsum_goo3DRes_shot_center.h5'
    dataset = h5py.File(dataset_path, 'a')
    
    tvsum_score_path = 'D:/Research/Video Summarization/Dataset/TvSum/tvsum_dataset/data/ydata-tvsum50-anno.tsv'
    table_path = '../datasets/Index_Name_Table_TVSum.txt'
    index_table = tools.read_index_table(table_path)
    name_table = tools.read_name_table(table_path)
    tvsum_score = tools.read_tvsum_score(tvsum_score_path)
    
    video_list = list(dataset.keys())
    
    for video_index in video_list:
        video_name = name_table[video_index]
        video = dataset[video_index]
        anno_scores = tvsum_score[video_name]
        segment = video['segmentation'][...]
        
        
        greedy_summaries = []
        for anno in anno_scores:
            summary = tools.generate_summary(anno, segment, 'greedy')
            greedy_summaries.append(np.array(summary))
        greedy_summaries = np.array(greedy_summaries)
        
        dataset[video_index]['greedy_summaries'] = greedy_summaries
        
        print(f'{video_index} Done...')
    
    dataset.close()
            
        
    
    

