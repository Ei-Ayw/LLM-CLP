import torch
import torch.nn as nn

# =============================================================================
# ### TextCNN 模型 ###
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

if __name__ == "__main__":
    model = TextCNN(vocab_size=1000)
    print("TextCNN model initialized.")
