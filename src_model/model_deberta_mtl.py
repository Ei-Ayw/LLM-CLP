import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config

# =============================================================================
# ### 改进模块 1：Attention Pooling (注意力池化层) ###
# 设计意图：
# 传统的 BERT 模型通常只使用 [CLS] 向量作为全局表示。
# 我们引入 Attention Pooling，让模型能够自动学习句子中每个 Token 的重要性权重，
# 从而更精准地捕捉到那些隐蔽的、具有攻击性的局部关键词特征。
# =============================================================================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # 学习每个 token 的权重得分
        self.score_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, last_hidden_state, attention_mask):
        # 1. 计算所有 token 的原始得分，shape: (batch_size, seq_len, 1)
        score = self.score_layer(last_hidden_state)
        
        # 2. Mask 处理：将填充（padding）部分的得分设为极小值，确保 softmax 后权重为 0
        if attention_mask is not None:
            score = score.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
            
        # 3. 通过 Softmax 归一化得到权重概率分布
        weights = torch.softmax(score, dim=1)
        
        # 4. 对所有 Token 的隐藏状态进行加权求和，得到池化后的向量 shape: (batch_size, hidden_size)
        weighted_sum = torch.sum(weights * last_hidden_state, dim=1)
        return weighted_sum

# =============================================================================
# ### 核心改进架构：DebertaToxicityMTL (多任务学习与公平性增强版) ###
# 基于 DeBERTa-V3 强大的语义编码能力，结合混合池化与多任务监督，全面提升模型的分类鲁棒性。
# =============================================================================
class DebertaToxicityMTL(nn.Module):
    def __init__(self, model_path_or_name="microsoft/deberta-v3-base", num_subtypes=6, num_identities=9, use_attention_pooling=True):
        super().__init__()
        # 加载 DeBERTa-V3 配置与预训练权重
        self.config = DebertaV2Config.from_pretrained(model_path_or_name, local_files_only=True)
        self.deberta = DebertaV2Model.from_pretrained(model_path_or_name, local_files_only=True)
        self.use_attention_pooling = use_attention_pooling
        
        hidden_size = self.config.hidden_size
        
        # ### 改进模块 2：Hybrid Pooling (混合特征提取) ###
        # 将 [CLS] 全局语义与 Attention Pooling 局部关键词特征拼接，形成更强有力的特征表示。
        if self.use_attention_pooling:
            self.pooling = AttentionPooling(hidden_size)
            self.projection = nn.Linear(hidden_size * 2, 512) # 拼接后维度翻倍 (768*2)
        else:
            self.projection = nn.Linear(hidden_size, 512)
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
        # ### 改进模块 3：Multi-Task Heads (多任务并行监督) ###
        # 通过增加子类型分类和身份预测任务，迫使模型学习更深层的社会属性与毒性特征的关联，
        # 而非单纯地记忆关键词。这是缓解模型偏见（如身份歧视）的核心架构设计。
        self.tox_head = nn.Linear(512, 1)           # 主任务：是否有毒 (Binary Classification)
        self.sub_head = nn.Linear(512, num_subtypes)   # 辅助任务1：毒性细分子类 (如侮辱、威胁)
        self.id_head = nn.Linear(512, num_identities)  # 辅助任务2：提及群体辨识 (如性别、宗教)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 1. 骨干网络特征提取
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # 2. 获取 [CLS] 全局向量（索引通常为 0）
        h_cls = last_hidden_state[:, 0, :]
        
        # 3. 混合池化策略实现
        if self.use_attention_pooling:
            # 执行注意力池化
            h_att = self.pooling(last_hidden_state, attention_mask)
            # ### 特征拼接：全局语义 + 局部敏感词语义 ###
            h = torch.cat([h_cls, h_att], dim=-1)
        else:
            h = h_cls
            
        # 4. 共享特征映射层 (Shared Bottleneck Layer)
        z = self.dropout(self.gelu(self.projection(h)))
        
        # 5. 输出多任务结果
        logits_tox = self.tox_head(z)
        logits_sub = self.sub_head(z)
        logits_id = self.id_head(z)
        
        return {
            "logits_tox": logits_tox,
            "logits_sub": logits_sub,
            "logits_id": logits_id,
            "features": z
        }

if __name__ == "__main__":
    # 模型初始化测试
    model = DebertaToxicityMTL("microsoft/deberta-v3-base")
    print("模型初始化成功，多任务架构已就绪。")
    
    # 模拟输入测试 (Batch=2, Seq_len=32)
    dummy_input_ids = torch.randint(0, 100, (2, 32))
    dummy_mask = torch.ones((2, 32))
    
    outputs = model(dummy_input_ids, dummy_mask)
    print("输出张量结构验证：")
    for k, v in outputs.items():
        print(f"任务项: {k:<12} | 形状: {list(v.shape)}")
