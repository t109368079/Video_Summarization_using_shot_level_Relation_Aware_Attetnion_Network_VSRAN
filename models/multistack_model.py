# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:09:09 2022

@author: gary
"""


import torch
import torch.nn as nn

from models.relation_aware_attention import RelationAwareAttention
from models.attention import Attention

class RAA_layer(nn.Module):
    def __init__(self, major_dim, minor_dim, hidden_dim, dropout_rate):
        super().__init__()
        
        self.major_dim = major_dim
        self.minor_dim = minor_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        
        self.attention_major = RelationAwareAttention(self.major_dim, self.minor_dim, self.hidden_dim, self.dropout_rate)
        self.attention_minor = RelationAwareAttention(self.minor_dim, self.major_dim, self.hidden_dim, self.dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
    
    def forward(self, x_major, x_minor):
        x_major = x_major.view(-1, self.major_dim)
        x_minor = x_minor.view(-1, self.minor_dim)
        
        y_major, _ = self.attention_major(x_major, x_minor)
        y_minor, _ = self.attention_minor(x_minor, x_major)
        
        del x_major
        del x_minor
        
        y_major = self.dropout(y_major)
        y_minor = self.dropout(y_minor)
        
        # Add Residual
        # y_major += x_major
        # y_minor += x_minor
        
        y_major = self.layer_norm(y_major)
        y_minor = self.layer_norm(y_minor)
    
        return y_major, y_minor
        

class MulitStack_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
    
        self.dimension_major = config.dimension_major
        self.dimension_minor = config.dimension_minor
        self.dimension_hidden = config.dimension_hidden
        self.dropout_rate = config.dropout_rate
        self.n_stack = config.n_stack
        
        # Mulit stack RelationAwareAttention
        self.initRAA = RAA_layer(self.dimension_major,self.dimension_minor,self.dimension_hidden,self.dropout_rate)
        self.hidden1RAA = RAA_layer(self.dimension_hidden, self.dimension_hidden, self.dimension_hidden, self.dropout_rate)
        # self.hidden2RAA = RAA_layer(self.dimension_hidden, self.dimension_hidden, self.dimension_hidden, self.dropout_rate)
        
        
        # Regression
        self.linear_a = nn.Linear(in_features = 2*self.dimension_hidden, out_features = 2*self.dimension_hidden)
        self.linear_b = nn.Linear(in_features = 2*self.dimension_hidden, out_features = 1, bias=False)
        
        self.layer_norm = nn.LayerNorm(2* self.dimension_hidden)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        return 
    
    def forward(self, X_major, X_minor):
        X_major = X_major.view(-1, self.dimension_major)
        X_minor = X_minor.view(-1, self.dimension_minor)
        
        Y_major, Y_minor = self.initRAA(X_major, X_minor)
        Y_major, Y_minor = self.hidden1RAA(Y_major, Y_minor)
        # Y_major, Y_minor = self.hidden2RAA(Y_major, Y_minor)
        
        Y = torch.cat((Y_major, Y_minor),1)
        
        del X_major, X_minor
        del Y_major, Y_minor

        
        # Regression
        Y = self.linear_a(Y)
        Y = self.relu(Y)
        Y = self.dropout(Y)
        Y = self.layer_norm(Y)
        
        Y = self.linear_b(Y)
        Y = self.sigmoid(Y)
        Y = Y.view(1,-1)
        
        return Y

if __name__ == '__main__':
    pass
        
        