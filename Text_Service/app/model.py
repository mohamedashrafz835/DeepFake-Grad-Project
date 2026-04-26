import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "aubmindlab/bert-base-arabertv02"

class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)

class AraBertCNNLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.bert.config.hidden_size  # 768

        self.conv3 = nn.Conv1d(hidden, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(hidden, 64, kernel_size=5, padding=2)
        self.relu  = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.pool = AttentionPool(256)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state  # (B, T, 768)

        xt = x.transpose(1, 2)
        c3 = self.relu(self.conv3(xt))
        c5 = self.relu(self.conv5(xt))

        x = torch.cat([c3, c5], dim=1).transpose(1, 2)
        lstm_out, _ = self.lstm(x)

        x = self.pool(lstm_out)

        return self.classifier(x)
