# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 16:44:24 2022

@author: Yuuki Misaki
"""

import copy
import torch
import torch.nn as nn
import numpy as np
import math as math

from models.layer import Simple_Layer

class My_Model(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.dim_major = config.dimension_major
        self.dim_minor = config.dimension_minor
        self.dim_hidden = config.dimension_hidden
        
        self.dropout_rate = config.dropout_rate
        self.N = config.n_stacks
        
        layer = Simple_Layer(config)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(self.N)])
        self.dropout_y = nn.Dropout(config.dropout_rate)
        self.layerNorm_y = nn.LayerNorm(self.dim_minor)
        self.linear_a = nn.Linear(in_features=self.dim_minor, out_features=self.dim_hidden)
        self.linear_b = nn.Linear(in_features=self.dim_hidden, out_features=1, bias=False)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layerNorm = nn.LayerNorm(self.dim_hidden)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
        return 
    
    def forward(self, X):
        X_major = X[0]
        X_minor = X[1]
        
        # Multi Stack Relation Aware Attention
        X_major = X_major.view(-1, self.dim_major)
        X_minor = X_minor.view(-1, self.dim_minor)
        for layer in self.layers:
            Y, A = layer(X_major, X_minor)

        Y = self.dropout_y(Y)
        Y = self.layerNorm_y(Y)
        
        # Regression
        Y = self.linear_a(Y)
        Y = self.relu(Y)
        Y = self.dropout(Y)
        Y = self.layerNorm(Y)
        
        Y = self.linear_b(Y)
        Y = self.sigmoid(Y)
        
        Y = Y.view(1, -1)
        
        return Y, A

if __name__ == "__main__":
    pass
        
