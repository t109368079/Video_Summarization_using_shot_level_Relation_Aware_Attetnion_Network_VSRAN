import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):

    def __init__(self, scaling_factor, dropout_rate):
        super().__init__()

        self.scaling_factor = scaling_factor
        self.dropout = nn.Dropout(dropout_rate)
        return


    def forward(self, Q, K, V, mask = None):
        # if number of heads = 2: Q (2 x m x dk), K (2 x n x dk), V (2 x n x dv)
        # if number of heads = 1: Q     (m x dk), K     (n x dk), V     (n x dv)

        # if Q, K, V are from the same sequence, m = n (self-attention)

        # S: similarities
        # if number of heads = 2: S (2 x m x n) = Q (2 x m x dk) · K.transpose (2 x dk x n)
        # if number of heads = 1: S     (m x n) = Q     (m x dk) · K.transpose     (dk x n)
        S = torch.matmul(Q * self.scaling_factor, K.transpose(-2, -1))

        if mask is not None:
            S = S.masked_fill(mask == 0, -1e9)

        # A: attention weights (scores)
        A = nn.functional.softmax(S, dim = -1)
        A = self.dropout(A)

        # Y: attention vectors
        # if #heads = 2:    Y (2 x m x dv) = A (2 x m x n) · V (n x dv)
        # if #heads = 1:    Y     (m x dv) = A     (m x n) · V (n x dv)
        Y = torch.matmul(A, V)
        return Y, A


if __name__ == '__main__':
    pass
