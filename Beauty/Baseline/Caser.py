import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import json
import numpy as np
import random
import os
import matplotlib.pyplot as plt

device = torch.device('cuda')

class Caser(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=2, num_items=0, max_seq_length=31, n_h=64, n_v=64, num_users=0):
        super(Caser, self).__init__()
        assert num_items > 100, "num_items is too small, please check the mapping dictionary or data"
        
        self.embedding = nn.Embedding(num_items, embedding_dim)
        
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=n_v, kernel_size=(max_seq_length, 1))
        
        self.conv_h = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_h, kernel_size=(i, embedding_dim)) for i in range(1, max_seq_length + 1)
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
        
        self.l1 = nn.Linear(4096, hidden_dim, bias=False)
        self.l2 = nn.Linear(4096, hidden_dim, bias=False)
        
    def forward(self, user, item_seq):

        item_seq_emb = item_seq.unsqueeze(1).contiguous() 
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

def prepare_data(file_path):
    users = []
    inputs = []
    inputs_cnt = []
    labels = []
    with open(file_path) as f:
        source_data = json.load(f)
    for key in source_data.keys():
        items = source_data[key][:-1]
        e_inputs = items[:-1]
        e_label = items[-1]
        users.append(key)
        inputs.append(pad_to_length(e_inputs, 31)) 
        inputs_cnt.append(len(e_inputs))        
        labels.append(e_label)                     
    return users, inputs, inputs_cnt, labels

def get_interests(u, icnt, semantic_long, semantic_short):

    user_keys = [f"user{str(au.item())}" for au in u]
    long_4096 = torch.concat([semantic_long[user_key].unsqueeze(0).to(device) for user_key in user_keys], dim=0)
    long_int = model.l2(long_4096)
    short_ints = []
    for xx, user_key in enumerate(user_keys):
        short_ints += [b.to(device).unsqueeze(0) for b in semantic_short[user_key]]
    short_ints = torch.concat(short_ints, dim=0)
    short_ints = model.l1(short_ints)
    return short_ints, long_int


def pad_to_length(seq, length=31):
    if len(seq) < length:
        return [0] * (length - len(seq)) + seq
    else:
        return seq[-length:]

def gumbel_sigmoid(logits, tau=1.0, hard=False):
    eps = 1e-10
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + eps) + eps)
    y = logits + g
    y = (y / tau).sigmoid()
    if hard:
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y
    return y

def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy().tolist()
    elif isinstance(tensor, np.ndarray):
        return tensor.tolist()
    return tensor

if __name__ == "__main__":
    random_seed = 50 

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)

    file_path = 'gru_item_list.json'
    users, inputs, inputs_cnt, labels = prepare_data(file_path)

    users = torch.tensor([int(u[4:]) for u in users])
    inputs = torch.tensor(inputs, dtype=torch.long)
    inputs_cnt = torch.tensor(inputs_cnt, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    num_items = 12102
    num_users = 22363

    dataset = TensorDataset(users, inputs, inputs_cnt, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Caser(embedding_dim=64, hidden_dim=128, num_layers=2, num_items=num_items, max_seq_length=31,
                  n_h=64, n_v=64, num_users=num_users).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_num}")
    print(f"Number of trainable parameters: {trainable_num}")

    num_epochs = 1000
    losses = []
    best_loss = np.inf
    patience = 10
    no_improvement_epochs = 0

    print("---------------begin training------------")
    model.train()

    semantic_long = torch.load('./semantic_long.pt')
    semantic_short = torch.load('./semantic_short.pt')

    from info_nce import InfoNCE
    infonce = InfoNCE()

    best_denoised = {} 

    for epoch in range(num_epochs):
        total_loss = 0
        epoch_mask_probs = {} 

        for u, x, icnt, y in dataloader:
            inputs_batch = x.to(device) 
            target = y.to(device)  
            icnt = icnt.to(device)
            u = u.to(device)
            
            optimizer.zero_grad()

            short_int, long_int = get_interests(u, icnt, semantic_long, semantic_short)
            
            inputs_embedded = model.embedding(inputs_batch)
            logit, hidden_seq, seq_output = model(u, inputs_embedded)
            loss = criterion(logit, target)

            gru_short_int = []
            for i, ic in enumerate(icnt):
                gru_short_int.append(hidden_seq[i, -ic:, :])
            gru_short_int = torch.concat(gru_short_int, 0)
            if short_int.shape[0] > gru_short_int.shape[0]:
                short_int = short_int[-gru_short_int.shape[0]:, :]
            gru_long_int = seq_output  

            cos_sim = F.cosine_similarity(gru_long_int, long_int, dim=-1)
            mask = cos_sim >= -1.0
            gru_slices, short_slices = [], []
            start_s = 0

            for i_user in range(inputs_embedded.size(0)):
                length_i = icnt[i_user].item()
                g_slice = hidden_seq[i_user, -length_i:, :]
                if g_slice.size(0) > 31:
                    g_slice = g_slice[-31:, :]
                s_slice = short_int[start_s:start_s+length_i, :]
                if s_slice.size(0) > 31:
                    s_slice = s_slice[-31:, :]
                min_len = min(g_slice.size(0), s_slice.size(0))
                g_slice = g_slice[-min_len:, :]
                s_slice = s_slice[-min_len:, :]
                gru_slices.append(g_slice)
                short_slices.append(s_slice)
                start_s += min_len

            mse_loss_total = 0.0

            for i_user in range(inputs_embedded.size(0)):
                user_id = u[i_user].item()
                long_i = long_int[i_user]  
                gru_long_i = seq_output[i_user]  
                out_i = gru_slices[i_user]     
                short_i = short_slices[i_user]   
                T = out_i.size(0)
                logits_each = torch.zeros(T, device=device)
                for j in range(T):
                    c1 = F.cosine_similarity(long_i.unsqueeze(0), out_i[j].unsqueeze(0), dim=-1)
                    c2 = F.cosine_similarity(gru_long_i.unsqueeze(0), short_i[j].unsqueeze(0), dim=-1)
                    c3 = F.cosine_similarity(out_i[j].unsqueeze(0), short_i[j].unsqueeze(0), dim=-1)
                    logits_each[j] = c1 + c2 + c3
                mask_probs = gumbel_sigmoid(logits_each, tau=1.0, hard=True)
                if user_id not in epoch_mask_probs:
                    epoch_mask_probs[user_id] = []
                epoch_mask_probs[user_id].append(mask_probs.cpu().tolist())

                denoised_out_i = out_i * mask_probs.unsqueeze(-1)
                recon_i = model.decoder(denoised_out_i)
                target_emb = inputs_embedded[i_user, -T:, :] 
                mse_i = F.mse_loss(recon_i, target_emb)
                mse_loss_total += mse_i

            loss = loss + mse_loss_total

            loss.backward()
            optimizer.step()

            print(f"Loss at this step: {loss.item():.4f}")
            losses.append(loss.item())
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement_epochs = 0
            savefile = "."
            torch.save(model.state_dict(), savefile)
            print(f"{savefile} saved with loss: {best_loss:.4f}")
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}")
            break
        