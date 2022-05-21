import torch
import torch.nn as nn

from models.relation_aware_attention import RelationAwareAttention
from models.attention import Attention

class Model(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dimension_major = config.dimension_major # dimension of CNN feature (primary)
        self.dimension_minor = config.dimension_minor # dimension of CNN feature (secondary)
        self.dimension_hidden = config.dimension_hidden
        self.dropout_rate = config.dropout_rate
        self.pe = config.pe

        self.attention_major = RelationAwareAttention(self.dimension_major, self.dimension_minor, self.dimension_hidden, self.dropout_rate, pe=self.pe, device=config.device)
        self.attention_minor = RelationAwareAttention(self.dimension_minor, self.dimension_major, self.dimension_hidden, self.dropout_rate, pe=self.pe, device=config.device)

        self.linear_a = nn.Linear(in_features = 2 * self.dimension_hidden, out_features = 2 * self.dimension_hidden)
        self.linear_b = nn.Linear(in_features = 2 * self.dimension_hidden, out_features = 1, bias = False)

        self.layer_normalization_y = nn.LayerNorm(self.dimension_hidden)
        self.layer_normalization_a = nn.LayerNorm(2 * self.dimension_hidden)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        return


    def forward(self, X_major, X_minor):
        X_major = X_major.view(-1, self.dimension_major)
        X_minor = X_minor.view(-1, self.dimension_minor)

        Y_minor, attention_weights = self.attention_major(X_major, X_minor)
        Y_major, attention_weights = self.attention_minor(X_minor, X_major)

        # residual
        # Y_major += X_major
        # Y_minor += X_minor

        Y_major = self.layer_normalization_y(Y_major)
        Y_minor = self.layer_normalization_y(Y_minor)

        Y = torch.cat((Y_major, Y_minor), 1)

        # regression
        Y = self.linear_a(Y)
        Y = self.relu(Y)
        Y = self.dropout(Y)
        Y = self.layer_normalization_a(Y)

        Y = self.linear_b(Y)
        Y = self.sigmoid(Y)
        Y = Y.view(1, -1)

        return Y, attention_weights


if __name__ == '__main__':
    pass
