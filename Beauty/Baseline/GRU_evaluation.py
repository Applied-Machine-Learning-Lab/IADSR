import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import torch.nn.functional as F
from info_nce import InfoNCE
device = torch.device('cuda')


class GRU4Rec(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, num_layers=2, num_items=0):
        super(GRU4Rec, self).__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.act = nn.Tanh()
        self.fc = nn.Linear(hidden_dim, num_items)
        self.l1 = nn.Linear(4096, hidden_dim, bias=False)
        self.l2 = nn.Linear(4096, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        out, h_n = self.gru(x)
        logit = self.fc(out[:, -1, :])
        logit = self.act(logit)
        return logit, out, None 


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

    model.eval()
    hr_list = []
    ndcg_list = []
    with torch.no_grad():
        for u, x, icnt, y in dataloader:
            inputs = x.to(device)
            target = y.to(device)
            inputs_embedded = model.embedding(inputs)
            logit, _, _ = model(inputs_embedded)
            _, topk_idx = torch.topk(logit, top_k, dim=-1)
            
            for i in range(target.size(0)):
                pred_list = topk_idx[i].cpu().numpy().tolist() 
                true_item = target[i].item()
                if true_item in pred_list:
                    hr_list.append(1)
                    rank = pred_list.index(true_item) 
                    ndcg_list.append(1.0 / np.log2(rank + 2))
                else:
                    hr_list.append(0)
                    ndcg_list.append(0.0)
    
    hr = np.mean(hr_list)
    ndcg = np.mean(ndcg_list)
    return hr, ndcg


def test_model(file_path, model_path, num_items, top_k=10):
    users, inputs, inputs_cnt, labels = prepare_data(file_path)
    users = torch.tensor([int(u[4:]) for u in users])
    inputs = torch.tensor(inputs, dtype=torch.long)
    inputs_cnt = torch.tensor(inputs_cnt, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(users, inputs, inputs_cnt, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = GRU4Rec(embedding_dim=64, hidden_dim=128, num_layers=2, num_items=num_items).to(device)
    model.load_state_dict(torch.load(model_path))
    
    hr, ndcg = evaluate_model(model, dataloader, top_k)
    print(f"Hit Rate (HR) @ {top_k}: {hr:.4f}")
    print(f"Normalized Discounted Cumulative Gain (NDCG) @ {top_k}: {ndcg:.4f}")


if __name__ == "__main__":
    file_path = './gru_item_list.json'
    model_path = '.' 
    num_items = 12102
    test_model(file_path, model_path, num_items, top_k=5)
    test_model(file_path, model_path, num_items, top_k=10)
    test_model(file_path, model_path, num_items, top_k=20)
