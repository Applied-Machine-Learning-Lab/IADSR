import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json

device = torch.device('cuda')

class Caser(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=2, num_items=0, max_seq_length=31, n_h=64, n_v=64, num_users=0):
        super(Caser, self).__init__()
        assert num_items > 100, "num_items is too small, please check the mapping dictionary or data"
        
        self.embedding = nn.Embedding(num_items, embedding_dim)
        
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=n_v, kernel_size=(max_seq_length, 1))
        
        self.conv_h = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_h, kernel_size=(i, embedding_dim))
            for i in range(1, max_seq_length + 1)
        ])
        
        self.fc1_dim_v = n_v * embedding_dim
        self.fc1_dim_h = n_h * max_seq_length
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU()

        self.fc_predict = nn.Linear(hidden_dim, num_items)

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        self.seq_conv = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=1)
        
        self.decoder = nn.Linear(hidden_dim, embedding_dim)
        
        self.l1 = nn.Linear(4096, hidden_dim, bias=False)
        self.l2 = nn.Linear(4096, hidden_dim, bias=False)
        
    def forward(self, user, item_seq):

        item_seq_emb = item_seq.unsqueeze(1)  
        user_emb = self.user_embedding(user)  

        out_v = self.conv_v(item_seq_emb) 
        out_v = out_v.view(-1, self.fc1_dim_v)  

        out_hs = []
        for conv in self.conv_h:
            conv_out = self.act(conv(item_seq_emb).squeeze(3))  
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) 
            out_hs.append(pool_out)
        out_h = torch.cat(out_hs, 1) 

        out = torch.cat([out_v, out_h], 1)
        out = self.dropout(out)

        z = self.act(self.fc1(out)) 
        x = torch.cat([z, user_emb], 1) 
        seq_output = self.act(self.fc2(x)) 

        logit = self.fc_predict(seq_output) 
        
        hidden_seq = item_seq.transpose(1, 2) 
        hidden_seq = self.seq_conv(hidden_seq) 
        hidden_seq = hidden_seq.transpose(1, 2)  
        
        return logit, hidden_seq, seq_output

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
    with open(file_path, 'r') as f:
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
            user = u.to(device)    
            
            inputs_embedded = model.embedding(inputs) 
            logit, _, _ = model(user, inputs_embedded) 
            
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

def evaluate_caser_model(file_path, model_path, num_items, num_users, top_k=10):
    users, inputs, inputs_cnt, labels = prepare_data(file_path)
    users = torch.tensor([int(u[4:]) for u in users])
    inputs = torch.tensor(inputs, dtype=torch.long)
    inputs_cnt = torch.tensor(inputs_cnt, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(users, inputs, inputs_cnt, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = Caser(embedding_dim=64, hidden_dim=128, num_layers=2, num_items=num_items, 
                  max_seq_length=31, n_h=64, n_v=64, num_users=num_users).to(device)
    model.load_state_dict(torch.load(model_path))
    
    hr, ndcg = evaluate_model(model, dataloader, top_k)
    print(f"Hit Rate (HR) @ {top_k}: {hr:.4f}")
    print(f"NDCG @ {top_k}: {ndcg:.4f}")

if __name__ == "__main__":
    file_path = '.'
    model_path = '.'
    num_items = 12102
    num_users = 22363

    evaluate_caser_model(file_path, model_path, num_items, num_users, top_k=20)
