import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import torch.nn.functional as F
import math

# 设置评估使用的设备
device = torch.device('cuda:3')

class SASRec(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_items, max_seq_length=31,n_heads=2, inner_size=100,
                 hidden_dropout_prob=0.5, attn_dropout_prob=0.5,
                 hidden_act='gelu', layer_norm_eps=1e-12, initializer_range=0.02):
        super(SASRec, self).__init__()
        assert num_items > 100, "num_items 过小，请检查映射字典或数据"
        self.embedding = nn.Embedding(num_items, embedding_dim)
        # 新增位置编码（序列最大长度取 31，与 pad_to_length 保持一致）
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.LayerNorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=n_heads,
            dim_feedforward=inner_size,
            dropout=attn_dropout_prob,
            activation=hidden_act
        )
        self.trm_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.proj = nn.Linear(embedding_dim, hidden_dim)
        
        self.act = nn.Tanh()
        self.fc = nn.Linear(hidden_dim, num_items)  # 输出层：预测下一个 item 的 ID
        
        # 以下参数保持不变，用于 InfoNCE 损失等
        self.l1 = nn.Linear(4096, hidden_dim, bias=False)
        self.l2 = nn.Linear(4096, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, embedding_dim)

        self.initializer_range = initializer_range  # 保存初始化参数
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        """
        x: shape (B, T, embedding_dim)，由外部先用 self.embedding 得到的嵌入表示
        返回:
         - logit: shape (B, num_items)，取序列最后位置经过 fc 层后并经过 Tanh 激活
         - out:   shape (B, T, hidden_dim)，Transformer 各时刻输出（经过投影）
         - output_tensor: shape (B, hidden_dim)，对 out 按时间维度做 max pooling 得到长期兴趣表示
        """
        B, T, D = x.size()
        # 添加位置编码
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos_emb = self.position_embedding(position_ids)
        x = x + pos_emb
        x = self.LayerNorm(x)
        x = self.dropout(x)
        
        # TransformerEncoder 内部要求输入维度为 (T, B, D)
        x = x.transpose(0, 1)  # 形状变为 (T, B, embedding_dim)
        trm_output = self.trm_encoder(x)  # (T, B, embedding_dim)
        trm_output = trm_output.transpose(0, 1)  # 恢复为 (B, T, embedding_dim)
        
        # 投影到 hidden_dim
        trm_output_proj = self.proj(trm_output)  # (B, T, hidden_dim)
        
        # 取序列最后一个位置作为预测依据（注意：由于 pad_to_length 的方式，最后一个位置一定是有效的）
        last_output = trm_output_proj[:, -1, :]  # (B, hidden_dim)
        logit = self.fc(last_output)  # (B, num_items)
        logit = self.act(logit)
        
        # 对 Transformer 输出做时间维度 max pooling，作为长期兴趣表示
        output_tensor, _ = torch.max(trm_output_proj, dim=1)  # (B, hidden_dim)
        
        return logit, trm_output_proj, output_tensor

# 数据预处理函数，与训练时一致
def pad_to_length(seq, length=31):
    if len(seq) < length:
        return [0] * (length - len(seq)) + seq
    else:
        return seq[-length:]

def prepare_data(file_path):
    users = []
    inputs = []
    inputs_cnt = []
    labels = []
    with open(file_path) as f:
        source_data = json.load(f)
    # 此处假设数据格式：每个用户对应一个 item 序列，最后一个为 label
    for key in source_data.keys():
        items = source_data[key]
        e_inputs = items[:-1]
        e_label = items[-1]
        users.append(key)
        inputs.append(pad_to_length(e_inputs, 31))
        inputs_cnt.append(len(e_inputs))
        labels.append(e_label)
    return users, inputs, inputs_cnt, labels

def evaluate_model(model, dataloader, top_k=10):
    """
    对每个样本：
      - 如果目标 item 出现在 top_k 预测中，则 HR 记为 1，
        同时根据目标 item 在列表中的位置计算 NDCG = 1/log₂(rank+2)（rank 从 0 开始）。
      - 否则，HR 和 NDCG 均为 0。
    """
    model.eval()
    hr_list = []
    ndcg_list = []
    with torch.no_grad():
        for u, x, icnt, y in dataloader:
            inputs = x.to(device)
            target = y.to(device)
            # 外部先调用 embedding 得到输入嵌入表示
            inputs_embedded = model.embedding(inputs)  # (B, T, embedding_dim)
            logit, _, _ = model(inputs_embedded)
            # 获取 top_k 个预测 item 的索引
            _, topk_idx = torch.topk(logit, top_k, dim=-1)
            
            # 对 batch 中每个样本计算 HR 与 NDCG
            for i in range(target.size(0)):
                pred_list = topk_idx[i].cpu().numpy().tolist()  # top-k 预测结果（列表）
                true_item = target[i].item()
                if true_item in pred_list:
                    hr_list.append(1)
                    rank = pred_list.index(true_item)  # 0-indexed 排名
                    ndcg_list.append(1.0 / np.log2(rank + 2))
                else:
                    hr_list.append(0)
                    ndcg_list.append(0.0)
    hr = np.mean(hr_list)
    ndcg = np.mean(ndcg_list)
    return hr, ndcg

def test_model(file_path, model_path, num_items, top_k=10):
    users, inputs, inputs_cnt, labels = prepare_data(file_path)
    # 此处假定用户 ID 格式如 "user123"，截取后部分转换为整数
    users = torch.tensor([int(u[4:]) for u in users])
    inputs = torch.tensor(inputs, dtype=torch.long)
    inputs_cnt = torch.tensor(inputs_cnt, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    dataset = TensorDataset(users, inputs, inputs_cnt, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 实例化 SASRec 模型，并加载训练好的模型参数
    model = SASRec(embedding_dim=64, hidden_dim=128, num_layers=2, num_items=num_items, max_seq_length=31,
                   n_heads=2, inner_size=100, hidden_dropout_prob=0.5, attn_dropout_prob=0.5,
                   hidden_act='gelu', layer_norm_eps=1e-12, initializer_range=0.02).to(device)
    model.load_state_dict(torch.load(model_path))
    
    hr, ndcg = evaluate_model(model, dataloader, top_k)
    print(f"Hit Rate (HR) @ {top_k}: {hr:.4f}")
    print(f"NDCG @ {top_k}: {ndcg:.4f}")

if __name__ == "__main__":
    # 测试数据路径及训练好的 SASRec 模型路径（请根据实际情况修改）
    file_path = '/home/tongzhouwu3/work/dataprocess/firstwork/GRU4Rec_denoised/all_denoised.json'
    model_path = '/home/tongzhouwu3/work/dataprocess/firstwork/SASREC_result/SASRec_all.pth'
    num_items = 12102  # 数据集中 item 的总数
    test_model(file_path, model_path, num_items, top_k=20)
