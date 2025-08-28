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

device = torch.device('cuda')

class SASRec(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_items, max_seq_length=31,n_heads=2, inner_size=100,
                 hidden_dropout_prob=0.5, attn_dropout_prob=0.5,
                 hidden_act='gelu', layer_norm_eps=1e-12, initializer_range=0.02):
        super(SASRec, self).__init__()
        assert num_items > 100, "num_items is too small, please check the mapping dictionary or data"
        self.embedding = nn.Embedding(num_items, embedding_dim)
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
        self.fc = nn.Linear(hidden_dim, num_items) 
        
        self.l1 = nn.Linear(4096, hidden_dim, bias=False)
        self.l2 = nn.Linear(4096, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, embedding_dim)

        self.initializer_range = initializer_range 
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

        B, T, D = x.size()
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos_emb = self.position_embedding(position_ids)
        x = x + pos_emb
        x = self.LayerNorm(x)
        x = self.dropout(x)
        
        x = x.transpose(0, 1) 
        trm_output = self.trm_encoder(x)
        trm_output = trm_output.transpose(0, 1) 
        
        trm_output_proj = self.proj(trm_output) 
        
        last_output = trm_output_proj[:, -1, :] 
        logit = self.fc(last_output)
        logit = self.act(logit)
        
        output_tensor, _ = torch.max(trm_output_proj, dim=1) 
        
        return logit, trm_output_proj, output_tensor

class SampledCrossEntropyLoss(nn.Module):
    def __init__(self, use_cuda):
        super(SampledCrossEntropyLoss, self).__init__()
        self.xe_loss = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda

    def forward(self, logit):
        batch_size = logit.size(1) 
        target = Variable(torch.arange(batch_size).long()).to(device)
        return self.xe_loss(logit, target)

def pad_to_length(seq, length=31):
    if len(seq) < length:
        return [0]*(length - len(seq)) + seq
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
        items = source_data[key][:-1]
        e_inputs = items[:-1]
        e_label = items[-1]
        users.append(key)
        inputs.append(pad_to_length(e_inputs,31)) 
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

    dataset = TensorDataset(users, inputs, inputs_cnt, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SASRec(embedding_dim=64, hidden_dim=128, num_layers=2, num_items=num_items, max_seq_length=31,
                   n_heads=2, inner_size=100, hidden_dropout_prob=0.5, attn_dropout_prob=0.5,
                   hidden_act='gelu', layer_norm_eps=1e-12, initializer_range=0.02).to(device)    
    criterion = SampledCrossEntropyLoss(use_cuda=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

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

    all_long_interest = []
    all_short_interest = []
    semantic_long = torch.load('./semantic_long.pt')
    semantic_short = torch.load('./semantic_short.pt')
    from info_nce import InfoNCE
    infonce = InfoNCE()

    user_mask_probs = {}

    for epoch in range(num_epochs):
        total_loss = 0

        for u, x, icnt, y in dataloader:
            inputs_embedded = model.embedding(x.to(device)) 
            target = y.to(device)
            icnt = icnt.to(device)
            u = u.to(device)
            
            optimizer.zero_grad()
            
            short_int, long_int = get_interests(u, icnt, semantic_long, semantic_short)
            logit, out, h_n = model(inputs_embedded)

            logit_sampled = logit[:, target.view(-1)]
            loss = criterion(logit_sampled)

            gru_short_int = []
            for i, ic in enumerate(icnt):
                gru_short_int.append(out[i, -ic: , :])
            gru_short_int = torch.concat(gru_short_int, 0)
            if short_int.shape[0] > gru_short_int.shape[0]:
                short_int = short_int[-gru_short_int.shape[0]: , :]

            gru_long_int = h_n
            infoloss = infonce(gru_short_int, short_int.to(device)) + infonce(gru_long_int, long_int.to(device))
            loss += infoloss

            cos_sim = F.cosine_similarity(gru_long_int, long_int, dim=-1)
            mask = cos_sim >= -1.0
            gru_slices, short_slices = [], []
            start_s = 0

            for i_user in range(inputs_embedded.size(0)):
                length_i = icnt[i_user].item()
                g_slice = out[i_user, -length_i:, :]

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
                long_i = long_int[i_user]  
                gru_long_i = gru_long_int[i_user]  
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

                user_id = u[i_user].item()
                if user_id not in user_mask_probs:
                    user_mask_probs[user_id] = []

                user_mask_probs[user_id].append(mask_probs.cpu().detach().numpy().tolist())

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
            print(f"{savefile} is saved!")
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}")
            break

    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.savefig('.')
    plt.close()
