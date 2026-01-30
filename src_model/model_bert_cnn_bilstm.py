import torch
import torch.nn as nn
from transformers import BertModel

class BertCNNBiLSTM(nn.Module):
    def __init__(self, model_name_or_path="bert-base-uncased", num_classes=1):
        super().__init__()
        try:
            # 移除 low_cpu_mem_usage=True 以解决 "Cannot copy out of meta tensor" 报错
            self.bert = BertModel.from_pretrained(model_name_or_path, use_safetensors=True)
        except Exception as e:
            print(f"\n❌ [CRITICAL] 无法加载 BertCNN 权重: {e}")
            print(f"👉 提示: 请确保已运行 'python download_models.py' 且下载完整。")
            raise e

        embedding_dim = self.bert.config.hidden_size
        self.conv = nn.Conv1d(embedding_dim, 256, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        x = last_hidden_state.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        x, _ = torch.max(lstm_out, dim=1)
        logits = self.fc(self.dropout(x))
        return {"logits_tox": logits}
