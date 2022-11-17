# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:28:49 2022

@author: Yuuki Misaki
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import h5py
import numpy as np
import os
import time

from models.model import My_Model
from trainer import Trainer
from utils.config import ConfigParser
from utils.kfold_split import generate_splits, read_splits
from utils.My_Dataset import My_VideoSummarizationDataset
from utils.My_Dataset import Aug_VideoSummarizationDataset
from utils.CoSum_DataLoader import CoSum_VideoSummarizationDataset


def write_report(text, path):
    with open(path, 'w') as fw:
        fw.write(text)
    return 

def save_score(score_dict, save_folder):
    for key in list(score_dict.keys()):
        save_path = os.path.join(save_folder,key+'.npy')
        score = score_dict[key]
        np.save(save_path,score)
    return 


if __name__ == '__main__':
    
    config_path = './augment_config.yaml'
    config = ConfigParser(config_path)
    
    #%% 讀取Augment Dataset
    aug_dataloader_list= []
    for dataset_path in config.augment_dataset_path:
        h5_aug_dataset = h5py.File(dataset_path, 'r')
        aug_dataset = Aug_VideoSummarizationDataset(h5_aug_dataset, config)
       
        indexes = [i for i in range(len(list(h5_aug_dataset.keys())))]
        sampler = SubsetRandomSampler(indexes)
        dataloader = DataLoader(aug_dataset, sampler=sampler)
        aug_dataloader_list.append(dataloader)
        
    
    model = My_Model(config)
    
    #%% 訓練一般資料集
   
    h5_test_dataset = h5py.File(config.test_dataset_path, 'r')
    
    if 'cosum' in config.test_dataset_path:
        test_dataset = CoSum_VideoSummarizationDataset(h5_test_dataset, config)
    else:
        test_dataset = My_VideoSummarizationDataset(h5_test_dataset, config)
    
    indexes = list(range(len(test_dataset)))
    if os.path.isfile(config.split_file_path):
        splits = read_splits(config.split_file_path)
        print("Read Split: ",end='')
    else:
        splits = generate_splits(indexes, number_of_splits=5, shuffle=True)
        print("Generate Split: ", end='')
    print(splits)
    splits_length = len(splits)
    
    average_f1_score_greedy = []
    average_f1_score_knapsack = []
    max_f1_score_greedy = []
    max_f1_score_knapsack = []
    max_kendall = []
    max_spearman = []
    max_mse = []
    max_mAP = []
    
    report_text = config.report_text+'\n'
    video_shot_score = {}
    for split_index, split in enumerate(splits):
        print(f'Fold {split_index + 1}/{splits_length}:')

        train_indexes = split['train']
        validate_indexes = split['validate']
        train_indexes = [i-1 for i in train_indexes]
        validate_indexes = [i-1 for i in validate_indexes]

        train_sampler = SubsetRandomSampler(train_indexes)
        validate_sampler = SubsetRandomSampler(validate_indexes)

        train_dataloader = DataLoader(test_dataset, batch_size = config.batch_size, sampler = train_sampler)
        validate_dataloader = DataLoader(test_dataset, batch_size = config.batch_size, sampler = validate_sampler)

        model = My_Model(config)
        
        # print(comment)
        
        trainer = Trainer(model, train_dataloader, validate_dataloader, config, split_index=split_index, aug_list=aug_dataloader_list)
        trainer.run()
        video_shot_score.update(trainer.video_shot_score_dict)
        

        average_f1_score_greedy.append(trainer.bsf_average_f1score_greedy)
        average_f1_score_knapsack.append(trainer.bsf_average_f1score_knapsack)
        max_f1_score_greedy.append(trainer.bsf_max_f1score_greedy)
        max_f1_score_knapsack.append(trainer.bsf_max_f1score_knapsack)
        max_kendall.append(trainer.kendall_record.get_max_epoch_mean())
        max_spearman.append(trainer.spearman_record.get_max_epoch_mean())
        max_mse.append(trainer.bsf_mse)
        max_mAP.append(trainer.bsf_mAP)
        report_text += f"Split: {split_index}, Best MSE: {trainer.bsf_mse} Best Kendall: {trainer.kendall_record.get_max_epoch_mean()}, Best Spearman: {trainer.spearman_record.get_max_epoch_mean()}, "
        report_text += f'Knapsack Average F1-score: {trainer.bsf_average_f1score_knapsack}, Max: {trainer.bsf_max_f1score_knapsack}, '
        report_text += f'Greesy Average F1-score: {trainer.bsf_average_f1score_greedy}, Max: {trainer.bsf_max_f1score_greedy}\n'
    
    average_f1_score_greedy_std = np.std(np.array(average_f1_score_greedy))
    average_f1_score_knapsack_std = np.std(np.array(average_f1_score_knapsack))
    max_f1_score_greedy_std = np.std(np.array(max_f1_score_greedy))
    max_f1_score_knapsack_std = np.std(np.array(max_f1_score_knapsack))
    
    kendall_std = np.std(np.array(max_kendall))
    spearman_std = np.std(np.array(max_spearman))
    mAP_std = np.std(np.array(max_mAP))
    
                                     
    
    average_f1_score_knapsack = np.mean(np.array(average_f1_score_knapsack))
    average_f1_score_greedy = np.mean(np.array(average_f1_score_greedy))
    max_f1_score_knapsack = np.mean(np.array(max_f1_score_knapsack))
    max_f1_score_greedy = np.mean(np.array(max_f1_score_greedy))
    
    report_text +=f'Knapsack Average F1-score: {average_f1_score_knapsack} ({average_f1_score_knapsack_std}), Max: {max_f1_score_knapsack} ({max_f1_score_knapsack_std})\n'
    report_text +=f'Greedy Average F1-score: {average_f1_score_greedy} ({average_f1_score_greedy_std}), Max: {max_f1_score_greedy} ({max_f1_score_greedy_std})\n'
    report_text +=f'Average Kendall: {np.mean(np.array(max_kendall))}({kendall_std})\n'
    report_text +=f'Average Spearman: {np.mean(np.array(max_spearman))}({spearman_std})\n'
    report_text +=f'Average MSE: {np.mean(np.array(max_mse))}\n'
    report_text +=f'Average mAP: {np.mean(np.array(max_mAP))}({mAP_std})'
    
    print(f'Knapsack Average F1-score: {average_f1_score_knapsack} ({average_f1_score_knapsack_std}), Max: {max_f1_score_knapsack} ({max_f1_score_knapsack_std})')
    print(f'Greedy Average F1-score: {average_f1_score_greedy} ({average_f1_score_greedy_std}), Max: {max_f1_score_greedy} ({max_f1_score_greedy_std})')
    print(f"Average Kendall score: {np.mean(np.array(max_kendall))}({kendall_std})")
    print(f"Average Spearman score: {np.mean(np.array(max_spearman))}({spearman_std})")
    print(f'Average mAP: {np.mean(np.array(max_mAP))}({mAP_std})')
    print(f'Average MSE: {np.mean(np.array(max_mse))}')
    write_report(report_text, config.report_path)
    save_score(video_shot_score, './save_score')
    
