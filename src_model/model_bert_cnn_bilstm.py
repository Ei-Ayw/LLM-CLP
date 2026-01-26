import torch
import torch.nn as nn
from transformers import BertModel

# =============================================================================
# ### 对比模型：BertCNNBiLSTM ###
# 设计说明：
# 这是一个经典的混合架构，用于验证在预训练模型基础上叠加传统时序处理模块的效果。
# 使用 CNN 捕捉局部 N-gram 特征，BiLSTM 捕捉双向长距离上下文，最后通过 Max Pooling 整合。
# =============================================================================
class BertCNNBiLSTM(nn.Module):
    """
    基于 BERT 预训练权重的 CNN-BiLSTM 混合架构模型。
    """
    def __init__(self, model_name_or_path="bert-base-uncased", num_classes=1):
        super().__init__()
        # 加载 BERT 骨干网络
        self.bert = BertModel.from_pretrained(model_name_or_path)
        embedding_dim = self.bert.config.hidden_size # 通常为 768
        
        # 1D-CNN：提取局部语义（卷积核大小为3，捕捉词与词之间的局部组合）
        self.conv = nn.Conv1d(embedding_dim, 256, kernel_size=3, padding=1)
        
        # 双向 LSTM：模拟人类阅读习惯，同时获取前向和后向语义
        self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        
        # 分类器
        self.fc = nn.Linear(128 * 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # 1. 语义特征提取（BERT）
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # 尺寸: (Batch, Seq, Hidden)
        
        # 2. 卷积处理 (需要调整张量维度以适配 Conv1d)
        # 输入: (B, H, L) -> 输出: (B, 256, L)
        x = last_hidden_state.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x.transpose(1, 2)  # 回到 (B, L, 256)
        
        # 3. 双向时序特征建模 (BiLSTM)
        lstm_out, _ = self.lstm(x)  # 输出尺寸: (B, L, 128*2)
        
        # 4. 全局池化 (Global Max Pooling)
        # 捕获整个句子中最显著的特征（即攻击性最强的部分）
        x, _ = torch.max(lstm_out, dim=1)
        
        # 5. 分类输出
        logits = self.fc(self.dropout(x))
        return {"logits_tox": logits}

if __name__ == "__main__":
    # 测试代码
    model = BertCNNBiLSTM()
    print(">>> BertCNNBiLSTM 模型初始化完成。")
    
    mock_ids = torch.randint(0, 500, (2, 32))
    mock_mask = torch.ones((2, 32))
    
    out = model(mock_ids, mock_mask)
    print(f">>> 输出形状: {list(out['logits_tox'].shape)}")
