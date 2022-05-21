import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

import utilities.video_summarization as utils
from utilities.metric import Metric
from utilities.visualization import plot

class Trainer():

    def __init__(self, model, train_dataloader, validate_dataloader, config, split_index):
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.config = config

        self.min_epochs = self.config.min_epochs
        self.max_epochs = self.config.max_epochs
        self.eval_rate = self.config.eval_rate
        self.split_index = split_index
        self.current_epoch = None

        self.model = model
        self.model.apply(self._initialize_weights)
        self.device = None
        self.bsf_model = None
        self.bsf_fscore = 0
        self.bsf_list = None
        self.bsf_kendall = 0

        self.optimizer = self._initialize_optimizer()
        self.criterion = nn.MSELoss()

        self.epoch_losses = Metric()        # training loss for epochs
        self.train_losses = Metric()        # training loss for each epoch
        self.test_losses = Metric()         # testing loss for each epoch
        self.epoch_test_losses = Metric()   # testing loss for epochs
        self.epoch_f1_scores = Metric()     # validation f1 score for epochs
        self.f1_scores = Metric()           # validation f1 score for each epoch
        self.kendall_scores = Metric()
        
        
        self.training_path = self.config.training_path
        self.testing_path = self.config.testing_path
        self.f1_score_path = self.config.f1_score_path
        self.report_text = ""
        return


    def _initialize_weights(self, module):
        if type(module) == nn.Linear:
            nn.init.xavier_uniform_(module.weight, gain = nn.init.calculate_gain('relu')) # gain = 1.414
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.1)
        return


    def _initialize_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config.learning_rate, weight_decay = self.config.l2_regularization)
        return optimizer


    # training loop
    def _train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        for DL in self.train_dataloader:
            for data, label, segmentation, summaries, key in tqdm(DL):
                data, label, segmentation, summaries, key = utils.preprocess(data, label, segmentation, summaries, key)
                
                major_feature = data[0]
                major_feature = major_feature.to(self.device)
                minor_feature = data[1]
                minor_feature = minor_feature.to(self.device)
                self.optimizer.zero_grad()
                output,_ = self.model(major_feature, minor_feature)
                label = label.to(self.device)
                loss = self.criterion(output, label)
                loss.backward()
                if self.device != torch.device('cpu'):
                    loss = loss.detach().cpu().numpy()
                else:
                    pass
                self.optimizer.step()
    
                self.train_losses.update(key, loss)
        return


    # validation loop
    def _validate(self):
        self.model.eval()
        with torch.no_grad():
            f1_list = {}
            for data, label, segmentation, summaries, key in tqdm(self.validate_dataloader):
                data, label, segmentation, summaries, key = utils.preprocess(data, label, segmentation, summaries, key)
                
                major_feature = data[0]
                major_feature = major_feature.to(self.device)
                minor_feature = data[1]
                minor_feature = minor_feature.to(self.device)
                label = label.to(self.device)
                output, attention_vector = self.model(major_feature, minor_feature)
                
                loss = self.criterion(output, label)
                if self.device != torch.device('cpu'):
                    loss = loss.detach().cpu().numpy()
                else:
                    pass

                output = output[0].cpu().numpy()                  # (1 x number of downsampled frames)
                attention_vector = attention_vector.cpu().numpy() # (number of downsampled frames x number of downsampled frames)
                
                if self.config.level == 'frame':
                    summary = utils.generate_summary(utils.upsample_sequence(output), segmentation)   #他原本用這個，但因為我現在是用shot所以暫時註解調
                else:
                    summary = utils.generate_summary_shot(output, segmentation)                       #如果要用generate summary based on shot的話，執行這行並註解上面那行
                # upsampled_frame_scores = []
                # for i in range(len(segmentation)):
                #     upsampled_frame_scores += [output[i]] * (segmentation[i][1] - segmentation[i][0] + 1)
                # summary = utils.generate_summary(upsampled_frame_scores, segmentation)
                precision, recall, f1_score = utils.evaluate_summary(summary, summaries, self.config.mode)
                
                gtscore = label.detach().cpu().numpy()
                corr = utils.eval_kendall(output, gtscore)
                
                self.test_losses.update(key, loss)
                self.f1_scores.update(key, f1_score)
                self.kendall_scores.update(key, corr)
                tmp = {key:f1_score}
                f1_list.update(tmp)
        return f1_list


    # 每個epoch結束時存檔
    def _save_model(self):
        if self.config.save_model:
            filename = f'{self.config.save_name}_{self.split_index}.pth'
            save_path = f'{self.config.checkpoints_path}/{filename}'
            torch.save(self.bsf_model.state_dict(), save_path)
            return
        else:
            return 

    def _on_training_start(self):
        self.device = torch.device(self.config.device)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.max_epochs}")
        print(f"Evaluate rate: {self.eval_rate}")
        print("=============Start Training===============")
        return


    def _on_epoch_start(self):
        self.train_losses.reset()
        self.f1_scores.reset()
        self.test_losses.reset()
        return


    def _on_epoch_end(self,f1_list):
        if (self.current_epoch+1) % self.eval_rate == 0:
            self.epoch_losses.update('', self.train_losses.mean)
            self.epoch_f1_scores.update('', self.f1_scores.mean)
            self.epoch_test_losses.update('',self.test_losses.mean)
            if self.epoch_f1_scores.current_value > self.bsf_fscore:
                self.bsf_model = self.model
                self.bsf_fscore = self.epoch_f1_scores.current_value
                self.bsf_list = f1_list
                self.bsf_kendall = self.kendall_scores.current_value
                self.bsf_test_loss = self.epoch_test_losses.current_value
            output_format = 'Epoch {} \t train loss = {:.6f} \t val test loss = {:.6f} \t F1 score (max) = {:.6f} ({:.6f}) \t kendall = {:.3f}'
            print(output_format.format(self.current_epoch + 1, self.epoch_losses.current_value, self.epoch_test_losses.current_value, self.epoch_f1_scores.current_value, self.epoch_f1_scores.max_value, self.kendall_scores.current_value))
        else:
            self.epoch_losses.update('', self.train_losses.mean)
            output_format = 'Epoch {} \t train loss = {:.6f}'
            print(output_format.format(self.current_epoch + 1, self.epoch_losses.current_value))
        # print(self.f1_scores.items)
        # self._save_model()
        return


    def _on_training_end(self):
        training_loss_hist = np.array(self.epoch_losses.values)
        testing_loss_hist = np.array(self.epoch_test_losses.values)
        f1_score_hist = np.array(self.epoch_f1_scores.values)
        
        training_path = self.training_path + '{}.npy'.format(self.split_index)
        testing_path = self.testing_path + '{}.npy'.format(self.split_index)
        f1_score_path = self.f1_score_path + '{}.npy'.format(self.split_index)
        
        np.save(training_path, training_loss_hist)
        np.save(testing_path, testing_loss_hist)
        np.save(f1_score_path, f1_score_hist)
        self.report_text = f"Split: {self.split_index}\t Best F-score: {self.bsf_fscore}\t Best Kendall: {self.bsf_kendall}\t MSE:{self.bsf_test_loss}"
        print("Successfully save training / testing result...")
        return


    def run(self):
        self._on_training_start()
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            self._on_epoch_start()
            self._train()
            f1_list = self._validate()
            self._on_epoch_end(f1_list)
        self._save_model()
        self._on_training_end()
        return self.bsf_list
    
    


if __name__ == '__main__':
    pass
