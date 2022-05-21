import torch
import torch.nn as nn
import math

class RelationAwareSelfAttention(nn.Module):

    def __init__(self, dm, dk, dv, dropout_rate):
        super().__init__()

        self.dm = dm # dm: dimension of model (feature dimension)
        self.dk = dk # dk: dimension of key (dq = dk)
        self.dv = dv # dv: dimension of value

        self.Wq = nn.Linear(in_features = self.dm, out_features = self.dk, bias = False) # Wq: weights for query
        self.Wk = nn.Linear(in_features = self.dm, out_features = self.dk, bias = False) # Wk: weights for key
        self.Wv = nn.Linear(in_features = self.dm, out_features = self.dv, bias = False) # Wv: weights for value
        self.Wo = nn.Linear(in_features = self.dv, out_features = self.dm, bias = False) # Wo: weights for output

        # self.scaling_factor = 0.06
        self.scaling_factor = 1 / math.sqrt(self.dk)
        self.dropout = nn.Dropout(dropout_rate)

        self.k = 10 # maximum relative distance
        self.wk = nn.Parameter(torch.Tensor(2 * self.k + 1, self.dm))
        torch.nn.init.xavier_uniform_(self.wk)

        self.max_length = 500
        self.table = self._create_table()
        return


    def _create_table(self):
        table = torch.LongTensor(self.max, self.max)
        for row in range(self.max):
            for column in range(self.max):
                if column - row > self.k:
                    table[row, column] = 2 * self.k
                elif column - row < -self.k:
                    table[row, column] = 0
                else:
                    table[row, column] = column - row + self.k
        return table


    def forward(self, X, mask = None):
        n = X.shape[0] # n: source length (sequence length)

        Q = self.Wq(X) # X (n x dm) -> Q (n x dk)
        K = self.Wk(X) # X (n x dm) -> K (n x dk)
        V = self.Wv(X) # X (n x dm) -> V (n x dv)

        # S: similarities
        S = torch.matmul(Q * self.scaling_factor, K.transpose(-2, -1)) # S (n x n) = Q (n x dk) · K.transpose (dk x n)


        Rk = self.wk[self.table]
        Rk = Rk[:n, :n]
        Rk = Rk.transpose(1, 2)                        # R (n x dk x n)
        Q = Q.view(n, 1, self.dk)                      # Q (n x 1 x dk)
        Sr = torch.matmul(Q * self.scaling_factor, Rk) # Sr (n x 1 x n) = Q (n x 1 x dk) · R (n x dk x n)
        Sr = Sr.view(n, n)                             # Sr (n x 1 x n) -> Sr (n x n)
        S = S + Sr


        if mask is not None:
            S = S.masked_fill(mask == 0, -1e9)

        # A: attention weights (scores)
        A = nn.functional.softmax(S, dim = -1)
        A = self.dropout(A)

        # Y: attention vectors
        Y = torch.matmul(A, V) # Y (n x dv) = A (n x n) · V (n x dv)

        Y = self.Wo(Y)
        return Y, A


if __name__ == '__main__':
    pass
