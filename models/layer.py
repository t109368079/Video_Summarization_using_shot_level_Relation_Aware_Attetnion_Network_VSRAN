# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 17:08:14 2022
Simple_Layer是簡化版 1.0.1ver(見圖片Baseline Model Ver1.0.1)
Res_Layer是Simple Layer加入feedforwar層與 Redisual Dropout(見圖片 Baseline Model Ver1.0.2)
(也就是把最原始的拿掉一半)


@author: Yuuki Misaki
"""

import torch
import torch.nn as nn

from models.relation_aware_attention import RelationAwareAttention

class Simple_Layer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.dim_major = config.dimension_major
        self.dim_minor = config.dimension_minor
        self.dim_hidden = config.dimension_hidden
        
        self.dropout_rate = config.dropout_rate
        self.k = config.max_relative_distance
        
        # Relation Aware Attention
        self.RAA = RelationAwareAttention(self.dim_major, self.dim_minor, self.dim_minor, self.dropout_rate, self.k, None, pe=config.pe, device=torch.device(config.device))
        self.RAA_dropout = nn.Dropout(self.dropout_rate)
        self.RAA_layerNorm = nn.LayerNorm(self.dim_minor)
        
               
        return
    
    def forward(self, X_major, X_minor):
        
        # Relation Aware Attention
        Y, Attent_weight = self.RAA(X_major, X_minor)
        Y = self.RAA_dropout(Y)
        
        # Layer Normalization
        Y = self.RAA_layerNorm(Y)
        return Y, Attent_weight
    
class Res_Layer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.dim_major = config.dimension_major
        self.dim_minor = config.dimension_minor
        self.dim_hidden = config.dimension_hidden
        
        self.dropout_rate = config.dropout_rate
        self.k = config.max_relative_distance
        
        # Relative Attention layer
        self.RAA = RelationAwareAttention(self.dim_major, self.dim_minor, self.dim_minor, self.dropout_rate, self.k, None, pe=config.pe, device=torch.device(config.device))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.RAA_layerNorm = nn.LayerNorm(self.dim_minor)
        
        # Feed Forward Layer
        self.ff1 = nn.Linear(in_features = self.dim_minor, out_features = self.dim_minor)
        self.ff2 = nn.Linear(in_features = self.dim_minor, out_features = self.dim_minor)
        self.layerNorm_ff = nn.LayerNorm(self.dim_minor)
        self.relu = nn.ReLU()
        return 
    
    def forward(self,X_major, X_minor):
        
        # Relative Attention
        Y, Attent_weight = self.RAA(X_major, X_minor)
        Y = self.dropout(Y)
        
        # Layer Normalization for RAA
        Y = self.RAA_layerNorm(Y)
        X_temp = Y
        
        # Feed Forward
        Y = self.ff1(Y)
        Y = self.relu(Y)
        Y = self.ff2(Y)
        
        # Residual dropout
        Y = self.dropout(Y)
        Y += X_temp
        
        # Layer Normalization for feed forward
        Y = self.layerNorm_ff(Y)
        
        return Y, Attent_weight
        
        
        
if __name__ == "__main__":
    pass
    

