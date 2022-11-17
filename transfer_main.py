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
    
    config_path = './transfer_config.yaml'
    config = ConfigParser(config_path)
    
    #%% 訓練Augment
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
    h5_train_dataset = h5py.File(config.train_dataset_path,'r')
    train_dataset = My_VideoSummarizationDataset(h5_train_dataset, config)
   
    h5_test_dataset = h5py.File(config.test_dataset_path, 'r')
    
    if 'cosum' in config.test_dataset_path:
        test_dataset = CoSum_VideoSummarizationDataset(h5_test_dataset, config)
    else:
        test_dataset = My_VideoSummarizationDataset(h5_test_dataset, config)
    
    video_f1_scores = []     # validation video f1 scores for each fold (5 x 10 or 5 x 5)
    
    
    report_text = config.report_text+'\n'
    video_shot_score = {}
    
    train_indexes = [i for i in range(len(list(h5_train_dataset.keys())))]
    test_indexes = [i for i in range(len(list(h5_test_dataset.keys())))]
    
    train_sampler = SubsetRandomSampler(train_indexes)
    validate_sampler = SubsetRandomSampler(test_indexes)
    
    train_dataloader = DataLoader(train_dataset, batch_size = config.batch_size, sampler = train_sampler)
    validate_dataloader = DataLoader(test_dataset, batch_size = config.batch_size, sampler = validate_sampler)


    trainer = Trainer(model, train_dataloader, validate_dataloader, config, aug_list=aug_dataloader_list)
    trainer.run()
    video_shot_score.update(trainer.video_shot_score_dict)
    result_table = trainer.bsf_result_table
        
    # report_text += f"Best F-score: {trainer.f1_scores.get_max_epoch_mean()}, Best MSE: {trainer.bsf_mse}, Best Kendall: {trainer.kendall_record.get_max_epoch_mean()}, Best Spearman: {trainer.spearman_record.get_max_epoch_mean()}\n"
    report_text += f'Knapsack Average F1-score: {trainer.bsf_average_f1score_knapsack} ({trainer.bsf_average_f1score_knapsack_std}), Max: {trainer.bsf_max_f1score_knapsack} ({trainer.bsf_max_f1score_knapsack_std})\n'
    report_text += f'Greedy Average F1-score: {trainer.bsf_average_f1score_greedy} ({trainer.bsf_average_f1score_greedy_std}), Max: {trainer.bsf_max_f1score_greedy} ({trainer.bsf_max_f1score_greedy_std})\n'
    
    report_text +=f'Average Kendall: {trainer.kendall_bsf_fscore} ({trainer.bsf_kendall_std})\n'
    report_text +=f'Average Spearman: {trainer.spearman_bsf_fscore} ({trainer.bsf_spearman_std})\n'
    report_text +=f'Average Test MSE: {trainer.bsf_mse}\n'
    report_text +=f'Average mAP(std): {trainer.bsf_mAP} ({trainer.bsf_mAP_std})'
    print(f'Knapsack Average F1-score: {trainer.bsf_average_f1score_knapsack} ({trainer.bsf_average_f1score_knapsack_std}), Max: {trainer.bsf_max_f1score_knapsack} ({trainer.bsf_max_f1score_knapsack_std})\n')
    print(f'Greedy Average F1-score: {trainer.bsf_average_f1score_greedy} ({trainer.bsf_average_f1score_greedy_std}), Max: {trainer.bsf_max_f1score_greedy} ({trainer.bsf_max_f1score_greedy_std})\n')
    print(f"Average Kendall score: {trainer.kendall_bsf_fscore} ({trainer.bsf_kendall_std})")
    print(f"Average Spearman score: {trainer.spearman_bsf_fscore} ({trainer.bsf_spearman_std})")
    print(f'Average Test MSE: {trainer.bsf_mse}')
    print(f'Average mAP (top-{config.mAP_top}): {trainer.bsf_mAP}({trainer.bsf_mAP_std})')
    write_report(report_text, config.report_path)
    save_score(video_shot_score, './save_score')
    result_table.to_csv(config.report_path.replace('txt','csv'))
    
