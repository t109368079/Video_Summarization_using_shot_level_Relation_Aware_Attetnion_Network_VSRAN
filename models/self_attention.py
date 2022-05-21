import torch
import torch.nn as nn
import math

from models.scaled_dot_product_attention import ScaledDotProductAttention

class SelfAttention(nn.Module):

    def __init__(self, dm, dk, dv, dropout_rate):
        super().__init__()

        self.dm = dm # dm: dimension of model (feature dimension)
        self.dk = dk # dk: dimension of key (dq = dk)
        self.dv = dv # dv: dimension of value

        self.Wq = nn.Linear(in_features = self.dm, out_features = self.dk, bias = False) # Wq: weights for query
        self.Wk = nn.Linear(in_features = self.dm, out_features = self.dk, bias = False) # Wk: weights for key
        self.Wv = nn.Linear(in_features = self.dm, out_features = self.dv, bias = False) # Wv: weights for value
        self.Wo = nn.Linear(in_features = self.dv, out_features = self.dm, bias = False) # Wo: weights for output

        self.scaling_factor = 0.06
        # self.scaling_factor = 1 / math.sqrt(self.dk)
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.scaling_factor, dropout_rate)
        return


    def forward(self, X, mask = None):
        n = X.shape[0] # n: source length (sequence length)

        Q = self.Wq(X) # X (n x dm) -> Q (n x dk)
        K = self.Wk(X) # X (n x dm) -> K (n x dk)
        V = self.Wv(X) # X (n x dm) -> V (n x dv)

        # Y: attention vectors
        # A: attention weights (scores)
        Y, A = self.scaled_dot_product_attention(Q, K, V, mask) # Y (n x dv), A (n x n)

        Y = self.Wo(Y)
        return Y, A


if __name__ == '__main__':
    pass
