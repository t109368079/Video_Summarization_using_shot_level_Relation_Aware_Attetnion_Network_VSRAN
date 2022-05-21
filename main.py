import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from torchsummary import summary

import os
import h5py
import numpy as np
import pandas as pd

from models.model import Model
from models.multistack_model import MulitStack_Model
from trainer import Trainer
from utilities.read_config import Read_split
from utilities.config import ConfigParser
from utilities.kfold_split import generate_splits
from utilities.video_summarization_dataset import VideoSummarizationDataset

if __name__ == '__main__':
    
    config_path = './config.yaml'
    config = ConfigParser(config_path)
    
    if config.Training_mode == 'Standard':
        h5_dataset = h5py.File(config.dataset_path, 'r')
        dataset = VideoSummarizationDataset(h5_dataset, config.label_name)
        
       
        indexes = list(range(len(dataset)))
        
        split_file_path = config.split_file_path
        if os.path.isfile(split_file_path):
            splits = Read_split(split_file_path)
            required_train_list = config.required_train_list
        else:
            splits = generate_splits(indexes, number_of_splits = 5, shuffle = True, config = config)
            required_train_list = [i for i in range(len(splits))]
        print(splits)
    
        splits_length = len(splits)
    
        f1_scores = []
        
        f1_list = {}
        report_text = config.report_text_title +'\n'
        
        for split_index in required_train_list:
            split = splits[split_index]
            print(f'Fold {split_index + 1}/{len(required_train_list)}:')
    
            train_indexes = split['train']
            validate_indexes = split['validate']
    
            train_sampler = SubsetRandomSampler(train_indexes)
            validate_sampler = SubsetRandomSampler(validate_indexes)
    
            train_dataloader = DataLoader(dataset, batch_size = config.batch_size, sampler = train_sampler)
            validate_dataloader = DataLoader(dataset, batch_size = config.batch_size, sampler = validate_sampler)
    
            model = Model(config)
    
            trainer = Trainer(model, [train_dataloader], validate_dataloader, config, split_index)
            tmp_dict = trainer.run()
            report_text += trainer.report_text
            report_text += '\n'
            
    
            f1_scores.append(trainer.epoch_f1_scores.max_value)
            f1_list.update(tmp_dict)
            print(tmp_dict)
    elif config.Training_mode == 'Transfer':
        required_train_list = [1]
        h5_train_dataset = h5py.File(config.dataset_path,'r')
        train_dataset = VideoSummarizationDataset(h5_train_dataset, config.train_label_name)
        
        h5_test_dataset = h5py.File(config.sub_dataset_path, 'r')
        test_dataset = VideoSummarizationDataset(h5_test_dataset, config.test_label_name)
        
        print(f'Training Dataset: {config.dataset_path}, Testing Dataset: {config.sub_dataset_path}')
        
        f1_scores = []
        f1_list = {}
        report_text = config.report_text_title +'\n'
        
        train_indexes = [i for i in range(len(list(h5_train_dataset.keys())))]
        test_indexes = [i for i in range(len(list(h5_test_dataset.keys())))]
        
        train_sampler = SubsetRandomSampler(train_indexes)
        validate_sampler = SubsetRandomSampler(test_indexes)

        train_dataloader = DataLoader(train_dataset, batch_size = config.batch_size, sampler = train_sampler)
        validate_dataloader = DataLoader(test_dataset, batch_size = config.batch_size, sampler = validate_sampler)

        model = Model(config)

        trainer = Trainer(model, [train_dataloader], validate_dataloader, config, 1)
        tmp_dict = trainer.run()
        report_text += trainer.report_text
        report_text += '\n'
        

        f1_scores.append(trainer.epoch_f1_scores.max_value)
        f1_list.update(tmp_dict)
        print(tmp_dict)
     
    elif config.Training_mode == "Augment":
        h5_train_dataset = h5py.File(config.dataset_path,'r')
        train_dataset = VideoSummarizationDataset(h5_train_dataset, config.train_label_name)
        
        h5_test_dataset = h5py.File(config.sub_dataset_path, 'r')
        test_dataset = VideoSummarizationDataset(h5_test_dataset, config.test_label_name)
        
        print(f'Training Dataset: {config.dataset_path}, Testing Dataset: {config.sub_dataset_path}')
        indexes = list(range(len(test_dataset)))
        
        split_file_path = config.split_file_path
        if os.path.isfile(split_file_path):
            splits = Read_split(split_file_path)
            required_train_list = config.required_train_list
        else:
            splits = generate_splits(indexes, number_of_splits = 5, shuffle = True, config = config)
            required_train_list = [i for i in range(len(splits))]
        print(splits)
    
        splits_length = len(splits)
    
        f1_scores = []
        
        f1_list = {}
        report_text = config.report_text_title +'\n'
        aug_indexes = [i for i in range(len(list(h5_train_dataset.keys())))]
        for split_index in required_train_list:
            split = splits[split_index]
            print(f'Fold {split_index + 1}/{len(required_train_list)}:')
    
            train_indexes = split['train']
            validate_indexes = split['validate']
    
            train_sampler = SubsetRandomSampler(train_indexes)
            aug_sampler = SubsetRandomSampler(aug_indexes)
            validate_sampler = SubsetRandomSampler(validate_indexes)
    
            train_dataloader = DataLoader(test_dataset, batch_size = config.batch_size, sampler = train_sampler)
            aug_dataloader = DataLoader(train_dataset, batch_size = config.batch_size, sampler = aug_sampler)
            validate_dataloader = DataLoader(test_dataset, batch_size = config.batch_size, sampler = validate_sampler)
    
            model = Model(config)
    
            trainer = Trainer(model, [train_dataloader, aug_dataloader], validate_dataloader, config, split_index)
            tmp_dict = trainer.run()
            report_text += trainer.report_text
            report_text += '\n'
            
    
            f1_scores.append(trainer.epoch_f1_scores.max_value)
            f1_list.update(tmp_dict)
            print(tmp_dict)     
        
    
    with open(config.report_path, 'w') as f:
        f.write(report_text)
    df = pd.DataFrame(f1_list,index=[0])
    df.to_csv('./f1_score.csv')
    print("Average F1: {}".format(sum(f1_scores)/len(required_train_list)))
    
    
