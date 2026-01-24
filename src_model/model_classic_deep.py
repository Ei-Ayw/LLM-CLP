import torch
import torch.nn as nn

# =============================================================================
# ### 1. TextCNN 模型 ###
# 设计说明：
# 利用多个不同大小的卷积核（3, 4, 5）并行提取句子中的局部 n-gram 特征。
# 能够有效捕捉到特定长度的攻击性短语。
# =============================================================================
class TextCNN(nn.Module):
    """
    经典的卷积神经网络文本分类模型。
    """
    def __init__(self, vocab_size, embed_dim=300, n_filters=100, filter_sizes=[3, 4, 5], output_dim=1, dropout=0.5):
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 多尺度卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, 
                      out_channels=n_filters, 
                      kernel_size=(fs, embed_dim)) 
            for fs in filter_sizes
        ])
        
        # 全连接层与正则化
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, attention_mask=None):
        # text: (batch_size, seq_len)
        embedded = self.embedding(text) # (batch_size, seq_len, embed_dim)
        embedded = embedded.unsqueeze(1) # (batch_size, 1, seq_len, embed_dim)
        
        # 卷积与最大池化
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved[n]: (batch_size, n_filters, seq_len - filter_sizes[n] + 1)
        
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled[n]: (batch_size, n_filters)
        
        # 特征拼接
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat: (batch_size, len(filter_sizes) * n_filters)
        
        return {"logits_tox": self.fc(cat)}

# =============================================================================
# ### 2. BiLSTM 模型 ###
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
    # 测试代码
    print(">>> 正在初始化经典深度学习模型...")
    cnn = TextCNN(vocab_size=1000)
    lstm = BiLSTM(vocab_size=1000)
    
    mock_input = torch.randint(0, 1000, (2, 32))
    out_cnn = cnn(mock_input)
    out_lstm = lstm(mock_input)
    
    print(f"TextCNN 输出形状: {list(out_cnn['logits_tox'].shape)}")
    print(f"BiLSTM  输出形状: {list(out_lstm['logits_tox'].shape)}")
