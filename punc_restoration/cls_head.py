from torch import nn
import torch


class ClassificationHead(nn.Module):
    def __init__(self, n_labels, hidden_size, do):
        super().__init__()

        # input shape (*, input_features)
        # output shape (*, output_features)
        self.dense = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(do)
        self.proj = nn.Linear(hidden_size, n_labels)

    def forward(self, batch):
        # batch.shape = [batch_size, max_seq_len, hidden_dim]
        batch = self.dropout(batch)

        batch = self.dense(batch)
        batch = torch.tanh(batch)
        batch = self.dropout(batch)

        result = self.proj(batch)
        return result
