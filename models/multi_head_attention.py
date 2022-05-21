import torch
import torch.nn as nn
import math

from models.scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):

    def __init__(self, h, dm, dk, dv, dropout_rate):
        super().__init__()

        self.h = h   # h: number of heads (number of parallel attention layers)

        self.dm = dm # dm: dimension of model (feature dimension)
        self.dk = dk # dk: dimension of key (dq = dk)
        self.dv = dv # dv: dimension of value

        self.Wq = nn.Linear(in_features = self.dm, out_features = self.h * self.dk, bias = False) # Wq: weights for expanding query
        self.Wk = nn.Linear(in_features = self.dm, out_features = self.h * self.dk, bias = False) # Wk: weights for expanding key
        self.Wv = nn.Linear(in_features = self.dm, out_features = self.h * self.dv, bias = False) # Wv: weights for expanding value
        self.Wo = nn.Linear(in_features = self.h * self.dv, out_features = self.dm, bias = False) # Wo: weights for output

        self.scaling_factor = 0.06
        # self.scaling_factor = 1 / math.sqrt(self.dk)
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.scaling_factor, dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)
        return


    def forward(self, Q, K, V, mask = None):
        n = K.shape[0] # n: source length
        m = Q.shape[0] # m: target length

        # if Q, K, V are from the same sequence, m = n (self-attention)

        expanded_Q = self.Wq(Q).view(m, self.h, self.dk) # Q (m x dk) -> expanded_Q (m x 2dk) -> expanded_Q (m x 2 x dk)
        expanded_K = self.Wk(K).view(n, self.h, self.dk) # K (n x dk) -> expanded_K (n x 2dk) -> expanded_K (n x 2 x dk)
        expanded_V = self.Wv(V).view(n, self.h, self.dv) # V (n x dv) -> expanded_V (n x 2dv) -> expanded_V (n x 2 x dv)

        expanded_Q = expanded_Q.transpose(0, 1) # expanded_Q.transpose (2 x m x dk)
        expanded_K = expanded_K.transpose(0, 1) # expanded_K.transpose (2 x n x dk)
        expanded_V = expanded_V.transpose(0, 1) # expanded_V.transpose (2 x n x dv)

        # Y: attention vectors
        # A: attention weights (scores)
        Y, A = self.scaled_dot_product_attention(expanded_Q, expanded_K, expanded_V, mask) # Y (2 x m x dv), A (2 x m x n)

        Y = Y.transpose(0, 1).contiguous().view(m, -1) # Y (2 x m x dv) -> Y (m x 2 x dv) -> Y (m x 2dv)
        Y = self.Wo(Y)                                 # Y (m x 2dv) -> Y (m x dv)
        Y = self.dropout(Y)
        Y += Q
        return Y, A


if __name__ == '__main__':
    pass
