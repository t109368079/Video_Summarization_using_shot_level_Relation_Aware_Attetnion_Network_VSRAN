# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 07:14:44 2022

@author: Yuuki Misaki
"""


import h5py
import numpy as np
from torch.utils.data import Dataset

class CoSum_VideoSummarizationDataset(Dataset):

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
        summaries = video['gt_summary'][...]
        user_score = video['gt_summary'][...]
        
        
        if len(summaries.shape) == 1:
            summaries = np.reshape(summaries, (1,summaries.shape[0]))
        return data, label, segmentation, summaries, key, indexes, np.reshape(user_score,(1,user_score.shape[0]))

    def __len__(self):
        return len(self.keys)
