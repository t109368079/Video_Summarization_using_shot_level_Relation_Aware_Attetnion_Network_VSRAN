"""
Date: 2022.04.17
Author: Lance
"""

import torch
import torch.nn as nn

import os
import numpy as np
import pandas as pd
from scipy.stats import kendalltau as kendall
from scipy.stats import spearmanr as spearman
from sklearn.metrics import average_precision_score as mAP
from matplotlib import pyplot as plt
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter

from utils.visualization import plot
from utils.loss_metric import LossMetric
from utils.f1_score_metric import F1ScoreMetric
import utils.video_summarization as utils

# 測試用
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.metrics import average_precision_score as mAP

from models.model import My_Model
from models.my_loss import My_loss
from utils.config import ConfigParser
from utils.kfold_split import generate_splits, read_splits
from utils.My_Dataset import My_VideoSummarizationDataset
from utils.earlyStop import EarlyStopping


class Trainer():

    def __init__(self, model, train_dataloader, validate_dataloader, config, aug_list=None, split_index=None):
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.aug_list = aug_list
        self.config = config
        self.split_index = split_index
        self.mode = self.config.mode
        self.device = torch.device(config.device)

        self.min_epochs = self.config.min_epochs
        self.max_epochs = self.config.max_epochs
        self.scheduler_metrics = []
        self.current_epoch = None
        self.earlyStop = EarlyStopping(delta=self.config.earlyStop_delta, patient=self.config.earlyStop_patient)
        self.model = model.to(self.device)
        self.model.apply(self._initialize_weights)


        self.optimizer,self.scheduler = self._initialize_optimizer()
        self.criterion = My_loss(config.alpha)

        self.training_com_losses = LossMetric()
        self.training_ken_losses = LossMetric()
        self.training_mse_losses = LossMetric()
        self.testing_com_losses = LossMetric()
        self.testing_ken_losses = LossMetric()
        self.testing_mse_losses = LossMetric()

        self.max_f1_scores_knapsack = F1ScoreMetric()
        self.average_f1_scores_knapsack = F1ScoreMetric()
        self.max_f1_scores_greedy = F1ScoreMetric()
        self.average_f1_scores_greedy = F1ScoreMetric()
        self.kendall_record = F1ScoreMetric()
        self.spearman_record = F1ScoreMetric()
        self.kendall_shot_record = F1ScoreMetric()
        self.spearman_shot_record = F1ScoreMetric()
        self.mAP_record = F1ScoreMetric()
        self.model_converge = False
        self.converge_epoch = None
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
        self.bsf_loss = 0
        self.bsf_mAP = 0
        self.bsf_mAP_std = 0
        self.bsf_model = None
        self.result_table = {}
        return


    def _initialize_weights(self, module):
        if type(module) == nn.Linear:
            nn.init.xavier_uniform_(module.weight, gain = nn.init.calculate_gain('relu')) # gain = 1.414
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.1)
        return


    def _initialize_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config.learning_rate, weight_decay = self.config.l2_regularization)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=2, verbose=True, threshold=0.00001, threshold_mode='rel', cooldown=30, min_lr=0, eps=1e-08)
        return optimizer,scheduler


    def check_model_converge(self, metric):
        cool_down = self.config.cool_down
        patient = self.config.patient
        threshold = self.config.threshold
        if self.current_epoch < cool_down:
            return 
        if len(self.scheduler_metrics) == 0:
            self.scheduler_metrics.append(metric)
        elif self.scheduler_metrics[0] > (1+threshold)*metric:
            self.scheduler_metrics = [metric]
        else:
            if len(self.scheduler_metrics) < patient:
                self.scheduler_metrics.append(metric)
            else:
                self.model_converge = True
                self.converge_epoch = self.current_epoch
        return 
        
        
        

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
                
                comb_loss, kendall_loss, mse_loss = self.criterion(output, label)
                comb_loss.backward()
                # print(loss.item())
                self.optimizer.step()
        return

    # training loop
    def _train(self):
        self.model.train()
        self.ken_loss_mean=[]
        self.mse_loss_mean=[]
        self.com_loss_mean=[]
        for data, label, segmentation, summaries, key , _, _  in tqdm(self.train_dataloader):
          data, label, segmentation, summaries, key, _ = utils.preprocess(data, label, segmentation, summaries, key)
          data = data[0].to(self.device), data[1].to(self.device)
          label = label.to(self.device)
          
          self.optimizer.zero_grad()
          # 下面這行是原本的
          # output, _ = self.model(data, segmentation, indexes, self.mode)
          # 這是新的
          output,_ = self.model(data)
        #   try:
        #       output,_ = self.model(data)
        #   except:
        #       # print(key)
        #       continue
          comb_loss, kendall_loss, mse_loss = self.criterion(output, label)
          comb_loss.backward()
          self.ken_loss_mean.append(kendall_loss.cpu().detach().item())
          self.mse_loss_mean.append(mse_loss.cpu().detach().item())
          self.com_loss_mean.append(comb_loss.cpu().detach().item())
          self.optimizer.step()
           
          self.training_com_losses.update(self.current_epoch, key, comb_loss.cpu().detach().numpy())
          tmp_dict = {'loss': comb_loss.cpu().detach().item(),'mse':mse_loss.cpu().detach().item(), 'kendall_loss':kendall_loss.cpu().detach().item(),
                       'shot_scores':output[0].cpu().detach().numpy()}
          self.epoch_training_table.update({key:tmp_dict})
        #   self.check_model_converge()
        epoch_loss,_ = self.training_com_losses.get_current_status()
        return


    # validation loop
    def _validate(self):
        self.model.eval()
        self.mse_loss_mean=[]
        self.ken_loss_mean=[]
        self.com_loss_mean=[]
        
        epoch_video_shot_score_dict = {}
        with torch.no_grad():
            average_score=[]
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
                comb_loss, kendall_loss, mse_loss = self.criterion(output, label)
                output = output[0].cpu().numpy()    # (1 x number of downsampled frames)
                label_np = label.detach().cpu().numpy()
                label_np = np.reshape(label_np, (1,-1))
                   

                # print(user_score.shape)
                kendall_coeff, spearman_coeff = utils.kendall_spearman(output, user_score, segmentation)
                kendall_shot, spearman_shot = utils.kendall_spearman(output, label_np)

                attention_weights = attention_weights.cpu().numpy() # (number of downsampled frames x number of downsampled frames)
                self.com_loss_mean.append(comb_loss.item())
                
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

                average_score.append([max_f1_score_knapsack, average_f1_score_knapsack, max_f1_score_greedy, average_f1_score_greedy, 
                                      kendall_coeff, spearman_coeff, kendall_shot, spearman_shot, mAP_score5, mAP_score15, mse_loss.item()])

                self.testing_com_losses.update(self.current_epoch, key, comb_loss.item())
                self.max_f1_scores_knapsack.update(self.current_epoch, key, max_f1_score_knapsack)
                self.average_f1_scores_knapsack.update(self.current_epoch, key, average_f1_score_knapsack)
                self.max_f1_scores_greedy.update(self.current_epoch, key, max_f1_score_greedy)
                self.average_f1_scores_greedy.update(self.current_epoch, key, average_f1_score_greedy)
                self.kendall_record.update(self.current_epoch, key, kendall_coeff)
                self.kendall_shot_record.update(self.current_epoch, key, kendall_shot)
                self.spearman_record.update(self.current_epoch, key, spearman_coeff)
                self.spearman_shot_record.update(self.current_epoch, key, spearman_shot)
                self.mAP_record.update(self.current_epoch, key, mAP_score)
                tmp = {'max_f1_scores_knapsack': max_f1_score_knapsack, 'average_f1_scores_knapsack': average_f1_score_knapsack, 
                        'max_f1_scores_greedy': max_f1_score_greedy, 'average_f1_scores_greedy': average_f1_score_greedy,
                        'kendall': kendall_coeff, 'spearman': spearman_coeff, 'kendall_shot': kendall_shot, 'spearman_shot': spearman_shot,
                        'mAP-5': mAP_score5, 'mAP-15': mAP_score15, 'mse': mse_loss.item(), 'kendall_loss': kendall_loss.item(),
                        'loss':comb_loss.item(),'shot_scores':output}
                self.epoch_result_table.update({key:tmp})                 
                epoch_video_shot_score_dict.update({key: output})
        average_score = np.array(average_score)
        average_score = sum(average_score)/len(average_score)
        epoch_info = {'video': key, 'max_f1_scores_knapsack': average_score[0], 'average_f1_scores_knapsack': average_score[1], 
        'max_f1_scores_greedy': average_score[2], 'average_f1_scores_greedy': average_score[3],
        'kendall': average_score[4], 'spearman': average_score[5], 'kendall_shot': average_score[6], 'spearman_shot': average_score[7],
        'mAP-5': average_score[8], 'mAP-15': average_score[9], 'mse': average_score[10]}
        return epoch_video_shot_score_dict, epoch_info

    def get_best_valid_video_seg_score(self):
        return self.video_shot_score_dict
    
    # 每個epoch結束時存檔
    def _save_model(self):
        save_path = self.config.save_model_path.format(self.config.dataset_name, self.config.dataset_path.split('/')[-1].replace('.h5', ''), self.split_index)
        torch.save(self.bsf_model.state_dict(), save_path)
        return


    def _delete_model(self, filename):
        file_path = f'{self.config.checkpoints_path}/{filename}'
        if os.path.isfile(file_path):
            os.remove(file_path)
        return


    def _on_training_start(self):
        return


    def _on_epoch_start(self):
        self.epoch_result_table = {}
        self.epoch_training_table = {}
        return


    def _on_epoch_end(self, tmp_video_shot_score_dict):
        
        # self.writer.add_scalar('Loss/train',statistics.mean(self.train_loss_mean), self.current_epoch)
        # self.writer.add_scalar('Loss/validation',statistics.mean(self.valid_loss_mean), self.current_epoch)
 
        # self.scheduler.step(statistics.mean(self.train_loss_mean)) 
        mean_loss,_ = self.training_com_losses.get_current_status()
        mean_test_loss, is_min_test_loss_update = self.testing_com_losses.get_current_status()

        if self.model_converge == False:
            self.check_model_converge(mean_test_loss)

        max_f1score_knapsack, max_max_f1score_knapsack, max_f1score_knapsack_std, is_max_f1score_knapsack_update = self.max_f1_scores_knapsack.get_current_status()
        average_f1score_knapsack, max_average_f1score_knapsack, average_f1score_knapsack_std, is_average_f1score_knapsack_update = self.average_f1_scores_knapsack.get_current_status()
        max_f1score_greedy, max_max_f1score_greedy, max_f1score_greedy_std, is_max_f1score_greedy_update = self.max_f1_scores_greedy.get_current_status()
        average_f1score_greedy, max_average_f1score_greedy, average_f1score_greedy_std, is_average_f1score_greedy_update = self.average_f1_scores_greedy.get_current_status()
        
        mean_kendall, max_kendall, kendall_std, is_max_kendall_update = self.kendall_record.get_current_status()
        mean_kendall_shot, max_kendall_shot, kendall_shot_std, is_max_kendall_shot_update = self.kendall_shot_record.get_current_status()
        mean_spearman, max_spearman, spearman_std, _ = self.spearman_record.get_current_status()
        mean_spearman_shot, max_spearman_shot, spearman_shot_std, _ = self.spearman_shot_record.get_current_status()
        mean_mAP, max_mAP, mAP_std, is_max_mAP_update = self.mAP_record.get_current_status()
        # self.writer.add_scalar('F1/validation',mean_f1_score, self.current_epoch)
        if self.config.is_verbose is True:
            output_format = 'Epoch {} \t train loss = {:.6f} \t library train loss = {:.6f} \t Kendall (max) = {:.6f} ({:.6f}) \t Spearman (max) = {:.6f} ({:.6f}) \t mAP (max) = {:.6f} ({:.6f})'
            print(output_format.format(self.current_epoch + 1, mean_loss, mean_test_loss, mean_kendall, max_kendall, mean_spearman, max_spearman, mean_mAP, max_mAP))
            print(f'Knapsack Max (bsf) {max_f1score_knapsack} ({self.bsf_max_f1score_knapsack}), Knapsack Average (bsf): {average_f1score_knapsack} ({self.bsf_average_f1score_knapsack})')
            print(f'Greedy Max (bsf) {max_f1score_greedy} ({self.bsf_max_f1score_greedy}), Greedy Average (bsf): {average_f1score_greedy} ({self.bsf_average_f1score_greedy})')
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
            raise AttributeError(f'Unrecognize argument master_metric: {self.config.master_metric_update}')

        epoch_dict = {'videos_result':self.epoch_result_table,'training_video_result':self.epoch_training_table, 'converge_epoch':self.converge_epoch}
        self.result_table.update({f'Epoch_{self.current_epoch}':epoch_dict})
        if master_metric_update and self.model_converge:                        # When master metric is update and model is converge then save result
            self.video_shot_score_dict = tmp_video_shot_score_dict
            self.bsf_max_f1score_knapsack = max_f1score_knapsack
            self.bsf_max_f1score_knapsack_std = max_f1score_knapsack_std
            self.bsf_average_f1score_knapsack = average_f1score_knapsack
            self.bsf_average_f1score_knapsack_std = average_f1score_knapsack_std
            self.bsf_max_f1score_greedy = max_f1score_greedy
            self.bsf_max_f1score_greedy_std = max_f1score_greedy_std
            self.bsf_average_f1score_greedy = average_f1score_greedy
            self.bsf_average_f1score_greedy_std = average_f1score_greedy_std
            self.bsf_model = self.model
            
            self.kendall_bsf_fscore = mean_kendall
            self.bsf_kendall_std = kendall_std
            self.spearman_bsf_fscore = mean_spearman
            self.bsf_spearman_std = spearman_std
            self.bsf_loss = mean_test_loss
            self.bsf_mAP = mean_mAP
            self.bsf_mAP_std = mAP_std
            if self.best_checkpoint is not None:
                self._delete_model(self.best_checkpoint)
        if is_max_kendall_update:
            self.bsf_kendall = mean_kendall
        return


    def _on_training_end(self):
        
        save_path = f'{self.config.checkpoints_path}/losses_{self.split_index}.csv'
        save_path_valid = f'{self.config.checkpoints_path}/valid_losses_{self.split_index}.csv'
        
        if self.config.save_model:
            print("Model saved")
            self._save_model()

        return


    def run(self):
        epochs_info = []
        self._on_training_start()
        for epoch in range(self.max_epochs):
            isConverge = self.model_converge
            self.current_epoch = epoch
            self._on_epoch_start()
            self._augment()
            self._train()
            if isConverge == False and self.model_converge:
                self.converge_epoch = epoch
            epoch_video_shot_score_dict, epoch_info = self._validate()
            epochs_info.append(epoch_info)
            if len(epoch_video_shot_score_dict) == 0:
                continue
            self._on_epoch_end(epoch_video_shot_score_dict)
            if self.earlyStop.__call__(self.testing_com_losses.get_current_status()[0]) and epoch > self.min_epochs:
                print("Stop training at epoch: {}".format(epoch))
                break
        self._on_training_end()
        self.epochs_info = epochs_info
        print(self.converge_epoch)
        return


if __name__ == '__main__':
    if None:
        print("None")
    else:
        print("lala")