import torch
import torch.nn as nn

# =============================================================================
# ### BiLSTM 模型 ###
# 设计说明：
# 使用双向长短期记忆网络捕捉长程语义依赖。
# 相比 CNN，BiLSTM 能够更好地理解句子的全局情绪和上下文逻辑。
# =============================================================================
class BiLSTM(nn.Module):
    """
    标准的双向 LSTM 文本分类模型。
    """
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256, output_dim=1, n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # BiLSTM 层
        self.lstm = nn.LSTM(embed_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=True, 
                           dropout=dropout,
                           batch_first=True)
        
        # 分类器
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, attention_mask=None):
        # text: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(text))
        
        # LSTM 处理
        output, (hidden, cell) = self.lstm(embedded)
        
        # 提取双向最后一个时刻的状态进行拼接
        # hidden 为 (num_layers * num_directions, batch_size, hidden_dim)
        hidden_last = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        return {"logits_tox": self.fc(self.dropout(hidden_last))}

if __name__ == "__main__":
    model = BiLSTM(vocab_size=1000)
    print("BiLSTM model initialized.")
