# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:19:18 2022

@author: Yuuki Misaki
"""

import h5py
import numpy as np
from torch.utils.data import Dataset

class My_VideoSummarizationDataset(Dataset):

    def __init__(self, dataset ,config):
        self.dataset = dataset
        self.config = config
        self.mode = config.mode
        self.keys = list(dataset.keys())
        return


    def __getitem__(self, index):
        key = self.keys[index]
        video = self.dataset[key]
        
        indexes=[]
        data = video[self.config.major_feature_name][...], video[self.config.minor_feature_name][...]
          
        label = video[self.config.gt_name][...]
        segmentation = video['segmentation'][...]
        summaries = video[self.config.user_summaries_name][...]
        
        try:
            user_score = video['anno_score'][...]
        except:
            user_score = video[self.config.user_summaries_name][...]
        
        
        
        if len(summaries.shape) == 1:
            summaries = np.reshape(summaries, (1,summaries.shape[0]))
        return data, label, segmentation, summaries, key, indexes, user_score

    def __len__(self):
        return len(self.keys)

class Aug_VideoSummarizationDataset(Dataset):
    
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.keys = list(dataset.keys())
    
    def __getitem__(self, index):
        key = self.keys[index]
        video = self.dataset[key]
        
        data = video[self.config.major_feature_name][...], video[self.config.minor_feature_name][...]
        label = video[self.config.gt_name][...]
        
        return data, label
    
    def __len__(self):
        return len(self.keys)

if __name__ == '__main__':
    path = '../datasets/tvsum_goo3DRes_shot_center.h5'
    dataset = h5py.File(path, 'r')
    video_name = list(dataset.keys())[0]
    video = dataset[video_name]
    
    
    print(video.keys())
