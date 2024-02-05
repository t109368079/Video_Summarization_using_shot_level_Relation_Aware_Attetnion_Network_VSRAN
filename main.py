import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import h5py
import random
import numpy as np
import pandas as pd
import os
import time
import pickle

from models.model import My_Model
from trainer import Trainer
from utils.config import ConfigParser
from utils.kfold_split import generate_splits, read_splits
from utils.video_summarization_dataset import VideoSummarizationDataset
from utils.My_Dataset import My_VideoSummarizationDataset

random.seed(777)
def write_report(text, path):
    with open(path, 'w') as fw:
        fw.write(text)
    return 

def save_score(score_dict, save_folder):
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    else:
        pass
    for key in list(score_dict.keys()):
        save_path = os.path.join(save_folder,key+'.npy')
        score = score_dict[key]
        np.save(save_path,score)
    return 

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

def gt_score_table(dataset, config):
    score_table = {}
    for video_index in list(dataset.keys()):
        video = dataset[video_index]
        gt_score = video[config.gt_name][...]
        score_table.update({video_index:gt_score})
    return score_table

if __name__ == '__main__':
    config_path = './config.yaml'
    config = ConfigParser(config_path)
    
    h5_dataset = h5py.File(config.dataset_path, 'r')
    dataset = My_VideoSummarizationDataset(h5_dataset, config)
   
    indexes = list(range(len(dataset)))
    if os.path.isfile(config.split_file_path):
        splits = read_splits(config.split_file_path)
        print("Read Split: ",end='')
    else:
        splits = generate_splits(indexes, number_of_splits = 5, shuffle = True)
        print("Generate Split: ",end='')
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
    
    #os.makedirs(config.checkpoints_path)
    
    report_text = config.report_text.format(config.dataset_name, config.shot_repre)
    report_text += f'Maximum relative distance: {config.max_relative_distance}\n'
    report_text += f'Number of stacks: {config.number_of_stacks}\n'
    video_shot_score = {}
    result_table = {}
    gt_score = gt_score_table(h5_dataset, config)
    result_table.update({'gt_score':gt_score})
    for split_index, split in enumerate(splits):
        print(f'Fold {split_index + 1}/{splits_length}:')

        train_indexes = split['train']
        validate_indexes = split['validate']
        train_indexes = [i-1 for i in train_indexes]
        validate_indexes = [i-1 for i in validate_indexes]

        train_sampler = SubsetRandomSampler(train_indexes)
        validate_sampler = SubsetRandomSampler(validate_indexes)

        train_dataloader = DataLoader(dataset, batch_size = config.batch_size, sampler = train_sampler)
        validate_dataloader = DataLoader(dataset, batch_size = config.batch_size, sampler = validate_sampler)

        model = My_Model(config)
        
        # print(comment)
        
        trainer = Trainer(model, train_dataloader, validate_dataloader, config, split_index=split_index, aug_list=None)
        trainer.run()
        video_shot_score.update(trainer.video_shot_score_dict)
        

        average_f1_score_knapsack.append(trainer.bsf_average_f1score_knapsack)
        average_f1_score_greedy.append(trainer.bsf_average_f1score_greedy)
        max_f1_score_knapsack.append(trainer.bsf_max_f1score_knapsack)
        max_f1_score_greedy.append(trainer.bsf_max_f1score_greedy)
        
        max_kendall.append(trainer.kendall_bsf_fscore)
        max_spearman.append(trainer.spearman_bsf_fscore)
        max_mse.append(trainer.bsf_loss)
        max_mAP.append(trainer.bsf_mAP)
        report_text += f"Split: {split_index},Best MSE: {trainer.bsf_loss}, Best Kendall: {trainer.kendall_record.get_max_epoch_mean()}, Best Spearman: {trainer.spearman_record.get_max_epoch_mean()}, Best mAP: {trainer.bsf_mAP}, "
        report_text += f"Knapsack Average F1-score: {trainer.bsf_average_f1score_knapsack}, Max F1-score: {trainer.bsf_max_f1score_knapsack}, "
        report_text += f'Greedy Average F1-scpre: {trainer.bsf_average_f1score_greedy}, Max F1-score: {trainer.bsf_max_f1score_greedy}\n'
        split_result_table = trainer.result_table
        result_table.update({f'Split_{split_index}':split_result_table})
        bsf_model = trainer.bsf_model
        if config.save_model:
            save_model(bsf_model, config.save_model_path.format(config.dataset_name, config.dataset_name, config.shot_repre, split_index))
        
    
    average_f1_score_knapsack_std = np.std(np.array(average_f1_score_knapsack))
    average_f1_score_greedy_std = np.std(np.array(average_f1_score_greedy))
    max_f1_score_knapsack_std = np.std(np.array(max_f1_score_knapsack))
    max_f1_score_greedy_std = np.std(np.array(max_f1_score_greedy))
    
    kendall_std = np.std(np.array(max_kendall))
    spearman_std = np.std(np.array(max_spearman))
    mAP_std = np.std(np.array(max_mAP))
    
    report_text +=f'Knapsack Average F1-score: {np.mean(np.array(average_f1_score_knapsack))}({average_f1_score_knapsack_std}), Max: {np.mean(np.array(max_f1_score_knapsack))}({max_f1_score_knapsack_std})\n'
    report_text +=f'Greedy Average F1-Score: {np.mean(np.array(average_f1_score_greedy))}({average_f1_score_greedy_std}), Max: {np.mean(np.array(max_f1_score_greedy))} ({max_f1_score_greedy_std})\n'
    report_text +=f'Average Kendall: {np.mean(np.array(max_kendall))}({kendall_std})\n'
    report_text +=f'Average Spearman: {np.mean(np.array(max_spearman))}({spearman_std})\n'
    report_text +=f'Average MSE: {np.mean(np.array(max_mse))}\n'
    report_text +=f'Average mAP: {np.mean(np.array(max_mAP))}({mAP_std})'
    print("")
    print(f'Knapsack Average F1-score: {np.mean(np.array(average_f1_score_knapsack))}({average_f1_score_knapsack_std}), Max: {np.mean(np.array(max_f1_score_knapsack))}({max_f1_score_knapsack_std})')
    print(f'Greedy Average F1-Score: {np.mean(np.array(average_f1_score_greedy))}({average_f1_score_greedy_std}), Max: {np.mean(np.array(max_f1_score_greedy))} ({max_f1_score_greedy_std})')
    print(f"Average Kendall score: {np.mean(np.array(max_kendall))}({kendall_std})")
    print(f"Average Spearman score: {np.mean(np.array(max_spearman))}({spearman_std})")
    print(f'Average MSE: {np.mean(np.array(max_mse))}')
    print(f'Average mAP(top-{config.mAP_top}): {np.mean(np.array(max_mAP))}({mAP_std})')
    write_report(report_text, config.report_path.format(config.dataset_name, config.dataset_name, config.shot_repre, config.user_summaries_name))
    save_score(video_shot_score, './save_score/{}'.format(config.dataset_name))
    
    # Write result to pickle file
    result_path = config.report_path.format(config.dataset_name, config.dataset_name, config.shot_repre, config.user_summaries_name).replace('.txt', '.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(result_table, f)
    
