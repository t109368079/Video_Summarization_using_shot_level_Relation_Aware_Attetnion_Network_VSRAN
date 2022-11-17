# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:51:57 2022

@author: Yuuki Misaki
"""

import os
import h5py
import random
import numpy as np

from utils import video_summarization as tools


def generate_random_score(n_segment):
    random_score = np.zeros(n_segment)
    
    for i in range(n_segment):
        random_score[i] = random.random()
    
    return random_score
    

if __name__ == '__main__':
    dataset_path = './datasets/summe_goo3DRes_shot_uni.h5'
    dataset = h5py.File(dataset_path, 'r')
    
    video_list = list(dataset.keys())
    mode = 'greedy'
    fscore_dict = {}
    avg_fscore = 0
    for video_name in video_list:
        video = dataset[video_name]
        
        segment = video['segmentation'][...]
        user_summaries = video['summaries'][...]
        
        n_seg = segment.shape[0]
        random_seg_score = generate_random_score(n_seg)
        random_summary = tools.generate_summary_shot(random_seg_score, segment, mode)
        _, _, fscore = tools.evaluate_summary(random_summary, user_summaries, metric='max')
        fscore_dict.update({video_name:fscore})
        
        if np.isnan(fscore):
            fscore = 0
        else:
            pass
        avg_fscore += fscore
    avg_fscore /= len(video_list)
    print("Average Fscore: {}".format(avg_fscore))
    
        
        
        
        
        
    



