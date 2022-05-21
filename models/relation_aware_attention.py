import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
    
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class RelationAwareAttention(nn.Module):

    def __init__(self, dm_major, dm_minor, dk, dropout_rate, dv = None, pe='rpe', device=torch.device('cuda:0')):
        super().__init__()

        self.dm_major = dm_major               # dm: dimension of major model (feature dimension)
        self.dm_minor = dm_minor               # dm: dimension of minor model (feature dimension)
        self.dk = dk                           # dk: dimension of key (dq = dk)
        self.dv = dv if dv is not None else dk # dv: dimension of value
        self.pe = pe

        self.Wq = nn.Linear(in_features = self.dm_major, out_features = self.dk, bias = False) # Wq: weights for query
        self.Wk = nn.Linear(in_features = self.dm_minor, out_features = self.dk, bias = False) # Wk: weights for key
        self.Wv = nn.Linear(in_features = self.dm_minor, out_features = self.dv, bias = False) # Wv: weights for value
        self.Wo = nn.Linear(in_features = self.dv, out_features = self.dv, bias = False)       # Wo: weights for output

        # self.scaling_factor = 0.06
        self.scaling_factor = 1 / math.sqrt(self.dk)
        self.dropout = nn.Dropout(dropout_rate)

        self.k = 10 # maximum relative distance
        self.wk = nn.Parameter(torch.Tensor(2 * self.k + 1, self.dk))
        torch.nn.init.xavier_uniform_(self.wk)

        self.max_length = 1500
        self.table = self._create_table()
        self.device = device
        return


    def _create_table(self):
        table = torch.LongTensor(self.max_length, self.max_length)
        for row in range(self.max_length):
            for column in range(self.max_length):
                if column - row > self.k:
                    table[row, column] = 2 * self.k
                elif column - row < -self.k:
                    table[row, column] = 0
                else:
                    table[row, column] = column - row + self.k
        return table


    def forward(self, X_major, X_minor):
        n = X_major.shape[0] # n: source length (sequence length)
        X_major = X_major.float()
        X_minor = X_minor.float()
        
        
        if self.pe == 'spe':
            PE_major = PositionalEncoding(X_major.shape[1]).to(self.device)
            PE_minor = PositionalEncoding(X_minor.shape[1]).to(self.device)
            
            X_major = PE_major(X_major)
            X_minor = PE_minor(X_minor)
        else:
            pass
        
        try:
            X_major.shape == X_minor.shape
        except:
            raise ValueError(f"major feature {X_major.shape} shape does not match minor feature {X_minor.shape} shape")
            
        
        Q = self.Wq(X_major) # X (n x dm) -> Q (n x dk)
        K = self.Wk(X_minor) # X (n x dm) -> K (n x dk)
        V = self.Wv(X_minor) # X (n x dm) -> V (n x dv)

        # S: similarities
        S = torch.matmul(Q * self.scaling_factor, K.transpose(-2, -1)) # S (n x n) = Q (n x dk) · K.transpose (dk x n)
       


        Rk = self.wk[self.table]
        Rk = Rk[:n, :n]
        Rk = Rk.transpose(1, 2)                        # R (n x dk x n)
        Q = Q.view(n, 1, self.dk)                      # Q (n x 1 x dk)
        Sr = torch.matmul(Q * self.scaling_factor, Rk) # Sr (n x 1 x n) = Q (n x 1 x dk) · R (n x dk x n)
        Sr = Sr.view(n, n)                             # Sr (n x 1 x n) -> Sr (n x n)
        # print(f"X_major shape: {X_major.shape}, X_minor shape: {X_minor.shape}")
        # print(f"S shape:{S.shape}, Sr shape: {Sr.shape}")
        if self.pe == 'rpe':    # Learned relative positional encoding
            S = S + Sr
        else:
            pass
        
        del Q, K, Sr
        

        # A: attention weights (scores)
        A = nn.functional.softmax(S, dim = -1)
        A = self.dropout(A)
        
        del S

        # Y: attention vectors
        Y = torch.matmul(A, V) # Y (n x dv) = A (n x n) · V (n x dv)

        
        Y = self.Wo(Y)
        return Y, A


if __name__ == '__main__':
    pass
