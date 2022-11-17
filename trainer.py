"""
Date: 2022.04.17
Author: Lance
"""

import torch
import torch.nn as nn

import os
import h5py
import numpy as np
import pandas as pd
from scipy.stats import kendalltau as kendall
from scipy.stats import spearmanr as spearman
from sklearn.metrics import average_precision_score as mAP
from matplotlib import pyplot as plt
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
import statistics

import utils.video_summarization as utils
from utils.loss_metric import LossMetric
from utils.f1_score_metric import F1ScoreMetric
from utils.visualization import plot

# 測試用
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.metrics import average_precision_score as mAP

from models.model import My_Model
from utils.config import ConfigParser
from utils.kfold_split import generate_splits, read_splits
from utils.My_Dataset import My_VideoSummarizationDataset


class Trainer():

    def __init__(self, model, train_dataloader, validate_dataloader, config, aug_list=None, split_index=None):
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.aug_list = aug_list
        self.config = config
        self.split_index = split_index
        self.mode = self.config.mode
        self.device = torch.device(config.device)

        self.max_epochs = self.config.epoches
        self.current_epoch = None
        self.model = model.to(self.device)
        self.model.apply(self._initialize_weights)


        self.optimizer,self.scheduler = self._initialize_optimizer()
        self.criterion = nn.MSELoss()
        
        

        self.losses = LossMetric()       # training loss for each epoch
        self.test_losses = LossMetric()  # Testing loss for each epoch
        self.max_f1_scores_knapsack = F1ScoreMetric()
        self.average_f1_scores_knapsack = F1ScoreMetric()
        self.max_f1_scores_greedy = F1ScoreMetric()
        self.average_f1_scores_greedy = F1ScoreMetric()
        self.kendall_record = F1ScoreMetric()
        self.spearman_record = F1ScoreMetric()
        self.mAP_record = F1ScoreMetric()
        self.train_loss_mean=[]
        self.valid_loss_mean=[]
        self.best_checkpoint = None
        self.video_shot_score_dict = {}
        self.bsf_fscore = 0
        self.kendall_bsf_fscore = 0
        self.spearman_bsf_fscore = 0
        self.bsf_max_f1score_knapsack = 0
        self.bsf_max_f1score_knapsack_std = 0
        self.bsf_average_f1score_knapsack = 0
        self.bsf_average_f1score_knapsack_std = 0
        self.bsf_max_f1score_greedy = 0
        self.bsf_max_f1score_greedy_std = 0
        self.bsf_average_f1score_greedy = 0
        self.bsf_average_f1score_greedy_std = 0
        self.bsf_mse = 0
        self.bsf_mAP = 0
        self.bsf_mAP_std = 0
        self.bsf_result_table = pd.DataFrame()
        return


    def _initialize_weights(self, module):
        if type(module) == nn.Linear:
            nn.init.xavier_uniform_(module.weight, gain = nn.init.calculate_gain('relu')) # gain = 1.414
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.1)
        return


    def _initialize_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config.learning_rate, weight_decay = self.config.l2_regularization)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        return optimizer,scheduler


    def _augment(self):
        if self.aug_list is None:
            return
        self.model.train()
        for aug_dataloader in self.aug_list:
            for data, label in tqdm(aug_dataloader):
                data, label = utils.aug_preprocess(data, label)
                data = data[0].to(self.device), data[1].to(self.device)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                try:
                    output, _ = self.model(data)
                except:
                    continue
                
                loss = self.criterion(output, label)
                loss.backward()
                # print(loss.item())
                
                self.optimizer.step()
        return

    # training loop
    def _train(self):
        self.model.train()
        self.train_loss_mean=[]
        for data, label, segmentation, summaries, key , _, _  in tqdm(self.train_dataloader):
          data, label, segmentation, summaries, key, _ = utils.preprocess(data, label, segmentation, summaries, key)
          data = data[0].to(self.device), data[1].to(self.device)
          label = label.to(self.device)
          
          self.optimizer.zero_grad()
          # 下面這行是原本的
          # output, _ = self.model(data, segmentation, indexes, self.mode)
          # 這是新的
          try:
              output,_ = self.model(data)
          except:
              # print(key)
              continue
          loss = self.criterion(output, label)
          loss.backward()
          self.train_loss_mean.append(loss.cpu().detach().item())
          self.optimizer.step()
          # self.scheduler.step(loss) 
          self.losses.update(self.current_epoch, key, loss.cpu().detach().numpy())
        return


    # validation loop
    def _validate(self):
        self.model.eval()
        self.valid_loss_mean=[]
        
        epoch_video_shot_score_dict = {}
        with torch.no_grad():
          for data, label, segmentation, summaries, key ,indexes, user_score in tqdm(self.validate_dataloader):
              data, label, segmentation, summaries, key, user_score = utils.preprocess(data, label, segmentation, summaries, key, user_score = user_score)

              data = data[0].to(self.device), data[1].to(self.device)
              label = label.to(self.device)
              # 這是原本的
              # output, attention_weights = self.model(data, segmentation, indexes,self.mode)
              # 這是新的
              try:
                  output, attention_weights = self.model(data)
              except:
                  continue
              loss = self.criterion(output, label)
              output = output[0].cpu().numpy()    # (1 x number of downsampled frames)
              label_np = label.detach().cpu().numpy()
              label_np = np.reshape(label_np, -1)
              kendall_coeff, spearman_coeff = utils.kendall_spearman(output, user_score, segmentation)
              # kendall_coeff,_ = kendall(output, label_np)
              # spearman_coeff, _ = spearman(output, label_np)
              attention_weights = attention_weights.cpu().numpy() # (number of downsampled frames x number of downsampled frames)
              self.valid_loss_mean.append(loss.item())
              
              if self.config.fixed_summary_length:
                  proportion = 0.15
              else:
                  proportion = utils.get_propotion(summaries)
              
              if True in np.isnan(output):
                  continue
              
              mAP_score5 = utils.mean_average_precision(output, summaries, segmentation, top=5)
              mAP_score15 = utils.mean_average_precision(output, summaries, segmentation, top=15)
              mAP_score = utils.mean_average_precision(output, summaries, segmentation, top=self.config.mAP_top)
              
              
              shot_summary_knapsack = utils.generate_summary_shot(output, segmentation, 'knapsack', proportion=proportion) # key_shot_selection are either knapsack or greedy
              shot_summary_greedy = utils.generate_summary_shot(output, segmentation, 'greedy', proportion=proportion)
              
              _, _, max_f1_score_knapsack = utils.evaluate_summary(shot_summary_knapsack, summaries, 'max')
              _, _, average_f1_score_knapsack = utils.evaluate_summary(shot_summary_knapsack, summaries, 'average')
              _, _, max_f1_score_greedy = utils.evaluate_summary(shot_summary_greedy, summaries, 'max')
              _, _, average_f1_score_greedy = utils.evaluate_summary(shot_summary_greedy, summaries, 'average')

              self.test_losses.update(self.current_epoch, key, loss.item())
              self.max_f1_scores_knapsack.update(self.current_epoch, key, max_f1_score_knapsack)
              self.average_f1_scores_knapsack.update(self.current_epoch, key, average_f1_score_knapsack)
              self.max_f1_scores_greedy.update(self.current_epoch, key, max_f1_score_greedy)
              self.average_f1_scores_greedy.update(self.current_epoch, key, average_f1_score_greedy)
              self.kendall_record.update(self.current_epoch, key, kendall_coeff)
              self.spearman_record.update(self.current_epoch, key, spearman_coeff)
              self.mAP_record.update(self.current_epoch, key, mAP_score)
              tmp = {'video': key, 'max_f1_scores_knapsack': max_f1_score_knapsack, 'average_f1_scores_knapsack': average_f1_score_knapsack, 
                     'max_f1_scores_greedy': max_f1_score_greedy, 'average_f1_scores_greedy': average_f1_score_greedy,
                     'kendall': kendall_coeff, 'spearman': spearman_coeff, 'mAP-5': mAP_score5, 'mAP-15': mAP_score15, 'mse': loss.item()}
              self.epoch_result_table = self.epoch_result_table.append(tmp, ignore_index=True)                     
              epoch_video_shot_score_dict.update({key: output})
        return epoch_video_shot_score_dict

    def get_best_valid_video_seg_score(self):
        return self.video_shot_score_dict
    
    # 每個epoch結束時存檔
    def _save_model(self, filename):
        save_path = f'{self.config.checkpoints_path}/{filename}'
        torch.save(self.model.state_dict(), save_path)
        self.best_checkpoint = filename
        return


    def _delete_model(self, filename):
        file_path = f'{self.config.checkpoints_path}/{filename}'
        if os.path.isfile(file_path):
            os.remove(file_path)
        return


    def _on_training_start(self):
        return


    def _on_epoch_start(self):
        self.epoch_result_table = pd.DataFrame()
        return


    def _on_epoch_end(self, tmp_video_shot_score_dict):
        
        # self.writer.add_scalar('Loss/train',statistics.mean(self.train_loss_mean), self.current_epoch)
        # self.writer.add_scalar('Loss/validation',statistics.mean(self.valid_loss_mean), self.current_epoch)
 
        # self.scheduler.step(statistics.mean(self.train_loss_mean)) 
        mean_loss,_ = self.losses.get_current_status()
        mean_test_loss, is_min_test_loss_update = self.test_losses.get_current_status()
        
        max_f1score_knapsack, max_max_f1score_knapsack, max_f1score_knapsack_std, is_max_f1score_knapsack_update = self.max_f1_scores_knapsack.get_current_status()
        average_f1score_knapsack, max_average_f1score_knapsack, average_f1score_knapsack_std, is_average_f1score_knapsack_update = self.average_f1_scores_knapsack.get_current_status()
        max_f1score_greedy, max_max_f1score_greedy, max_f1score_greedy_std, is_max_f1score_greedy_update = self.max_f1_scores_greedy.get_current_status()
        average_f1score_greedy, max_average_f1score_greedy, average_f1score_greedy_std, is_average_f1score_greedy_update = self.max_f1_scores_greedy.get_current_status()
        
        mean_kendall, max_kendall, kendall_std, is_max_kendall_update = self.kendall_record.get_current_status()
        mean_spearman, max_spearman, spearman_std, _ = self.spearman_record.get_current_status()
        mean_mAP, max_mAP, mAP_std, is_max_mAP_update = self.mAP_record.get_current_status()
        # self.writer.add_scalar('F1/validation',mean_f1_score, self.current_epoch)
        if self.config.is_verbose is True:
            output_format = 'Epoch {} \t train loss = {:.6f} \t library train loss = {:.6f} \t Kendall (max) = {:.6f} ({:.6f}) \t Spearman (max) = {:.6f} ({:.6f}) \t mAP (max) = {:.6f} ({:.6f})'
            print(output_format.format(self.current_epoch + 1, mean_loss, mean_test_loss, mean_kendall, max_kendall, mean_spearman, max_spearman, mean_mAP, max_mAP))
            print(f'Knapsack Max (bsf) {max_f1score_knapsack} ({max_max_f1score_knapsack}), Knapsack Average (bsf): {average_f1score_knapsack} ({max_average_f1score_knapsack})')
            print(f'Greedy Max (bsf) {max_f1score_greedy} ({max_max_f1score_greedy}), Greedy Average (bsf): {average_f1score_greedy} ({max_average_f1score_greedy})')
            # print(self.f1_scores.get_epoch(self.current_epoch))
            
        if self.config.master_metric == 'f1_score':
            # 這邊很麻煩，要先看要Knapsack 還是 Greedy，再看選平均還是最大
            if self.config.key_shot_selection == 'knapsack':
                if self.config.eval_mode == 'max':
                    master_metric_update = is_max_f1score_knapsack_update
                elif self.config.eval_mode == 'average':
                    master_metric_update = is_average_f1score_knapsack_update
            else:
                if self.config.eval_mode == 'max':
                    master_metric_update = is_max_f1score_greedy_update
                elif self.config.eval_mode == 'average':
                    master_metric_update = is_average_f1score_greedy_update
        elif self.config.master_metric == 'kendall':
            master_metric_update = is_max_kendall_update
        elif self.config.master_metric == 'mAP':
            master_metric_update = is_max_mAP_update
        elif self.config.master_metric == 'MSE':
            master_metric_update = is_min_test_loss_update
        else:
            raise AttributeError(f'Unrecognize argument master_metric: {config.master_metric_update}')
            
        if master_metric_update:
            self.video_shot_score_dict = tmp_video_shot_score_dict
            
            self.bsf_max_f1score_knapsack = max_f1score_knapsack
            self.bsf_max_f1score_knapsack_std = max_f1score_knapsack_std
            self.bsf_average_f1score_knapsack = average_f1score_knapsack
            self.bsf_average_f1score_knapsack_std = average_f1score_knapsack_std
            self.bsf_max_f1score_greedy = max_f1score_greedy
            self.bsf_max_f1score_greedy_std = max_f1score_greedy_std
            self.bsf_average_f1score_greedy = average_f1score_greedy
            self.bsf_average_f1score_greedy_std = average_f1score_greedy_std
            
            self.kendall_bsf_fscore = mean_kendall
            self.bsf_kendall_std = kendall_std
            self.spearman_bsf_fscore = mean_spearman
            self.bsf_spearman_std = spearman_std
            self.bsf_mse = mean_test_loss
            self.bsf_mAP = mean_mAP
            self.bsf_mAP_std = mAP_std
            self.bsf_result_table = self.epoch_result_table
            if self.best_checkpoint is not None:
                self._delete_model(self.best_checkpoint)
            # filename = 'epoch_{}_{}_{:.4f}.pth'.format(self.split_index, self.current_epoch, mean_f1_score)
            # self._save_model(filename)
        if is_max_kendall_update:
            self.bsf_kendall = mean_kendall
        return


    def _on_training_end(self):
        
        save_path = f'{self.config.checkpoints_path}/losses_{self.split_index}.csv'
        save_path_valid = f'{self.config.checkpoints_path}/valid_losses_{self.split_index}.csv'
        
        # Plot Training and testing loss
        training_loss = self.losses.get_epoch_means()
        testing_loss = self.test_losses.get_epoch_means()
        plt.figure()
        plt.plot(training_loss, label='Training')
        plt.plot(testing_loss, label='Testing')
        plt.legend()
        plt.show()
        
        # self.losses.get_epoch_means().to_csv(save_path)
        # self.losses.get_valid_epoch_means().to_csv(save_path_valid)
        # # files.download(save_path)
        # files.download(save_path_valid)
        return


    def run(self):
        self._on_training_start()
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            self._on_epoch_start()
            self._augment()
            self._train()
            tmp = self._validate()
            if len(tmp) == 0:
                continue
            self._on_epoch_end(tmp)
        self._on_training_end()
        return


if __name__ == '__main__':
    if None:
        print("None")
    else:
        print("lala")