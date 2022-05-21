import torch
import torch.nn as nn
import math

from models.scaled_dot_product_attention import ScaledDotProductAttention

class Attention(nn.Module):

    def __init__(self, dm_major, dm_minor, dk, dropout_rate, dv = None):
        super().__init__()

        self.dm_major = dm_major               # dm: dimension of major model (feature dimension)
        self.dm_minor = dm_minor               # dm: dimension of minor model (feature dimension)
        self.dk = dk                           # dk: dimension of key (dq = dk)
        self.dv = dv if dv is not None else dk # dv: dimension of value

        self.Wq = nn.Linear(in_features = self.dm_major, out_features = self.dk, bias = False) # Wq: weights for query
        self.Wk = nn.Linear(in_features = self.dm_minor, out_features = self.dk, bias = False) # Wk: weights for key
        self.Wv = nn.Linear(in_features = self.dm_minor, out_features = self.dv, bias = False) # Wv: weights for value
        self.Wo = nn.Linear(in_features = self.dv, out_features = self.dv, bias = False)       # Wo: weights for output

        # self.scaling_factor = 0.06
        self.scaling_factor = 1 / math.sqrt(self.dk)
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.scaling_factor, dropout_rate)
        return


    def forward(self, X_major, X_minor):
        n = X_major.shape[0] # n: source length (sequence length)

        Q = self.Wq(X_major) # X (n x dm) -> Q (n x dk)
        K = self.Wk(X_minor) # X (n x dm) -> K (n x dk)
        V = self.Wv(X_minor) # X (n x dm) -> V (n x dv)

        # Y: attention vectors
        # A: attention weights (scores)
        Y, A = self.scaled_dot_product_attention(Q, K, V) # Y (n x dv), A (n x n)

        Y = self.Wo(Y)
        return Y, A


if __name__ == '__main__':
    pass
