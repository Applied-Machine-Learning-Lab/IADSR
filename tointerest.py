import os
import time
import json
from collections import defaultdict
import torch
from tqdm import tqdm
from llm2vec import LLM2Vec
from transformers import AutoTokenizer, AutoModel

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

print("Loading semantic data...")
with open('semantic.json', 'r', encoding='utf-8') as f:
    semantic_data = json.load(f)

model_path = "./"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Loading model...")
model = AutoModel.from_pretrained(model_path).cuda()
print("Model loaded.")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

llm2vec_model = LLM2Vec(model=model, tokenizer=tokenizer)

def batch_encode_texts(texts, batch_size=512):
    all_embeds = []
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            end_idx = start_idx + batch_size
            batch_texts = texts[start_idx:end_idx]
            
            batch_embeds = llm2vec_model.encode(batch_texts)
            all_embeds.append(batch_embeds.detach().cpu())

    if len(all_embeds) > 1:
        return torch.cat(all_embeds, dim=0)
    elif all_embeds:
        return all_embeds[0]
    else:
        return torch.empty(0)

def process_semantic_data(data):
    all_texts = set()
    
    user_sequences = {}
    user_short_texts = {}
    user_long_texts = {}
    users_to_skip = []
    
    for user, items in tqdm(data.items(), desc="Collecting user sequences"):
        valid_items = [item.strip() for item in items if item.strip() and item.strip() != '[PAD]']
        
        if len(valid_items) < 2:
            users_to_skip.append(user)
            continue
        
        user_sequences[user] = valid_items
        
        short_texts = []
        
        for end in range(1, len(valid_items) - 1): 
            sequence = valid_items[:end]
            text = ", ".join(sequence)
            short_texts.append(text)
            all_texts.add(text)
        
        long_sequence = valid_items[:len(valid_items)-1]
        long_text = ", ".join(long_sequence)
        all_texts.add(long_text)
        
        user_short_texts[user] = short_texts
        user_long_texts[user] = long_text
    
    print(f"Total of {len(all_texts)} different texts need to be encoded")
    
    all_texts_list = list(all_texts)
    
    print("Encoding all texts...")
    all_embeds = batch_encode_texts(all_texts_list, batch_size=500)
    print(f"Encoding completed, embedding dimension: {all_embeds.shape}")
    
    text_to_embed = {text: all_embeds[i] for i, text in enumerate(all_texts_list)}
    
    short_term_embeddings = {}
    long_term_embeddings = {}
    
    for user in tqdm(user_sequences.keys(), desc="Creating user embeddings"):
        if user in user_short_texts and user in user_long_texts:
            short_embeds = [text_to_embed[text] for text in user_short_texts[user]]
            
            long_embed = text_to_embed[user_long_texts[user]]
            
            short_term_embeddings[user] = short_embeds
            long_term_embeddings[user] = long_embed
    
    return short_term_embeddings, long_term_embeddings, users_to_skip

start_time = time.time()
print("Start processing semantic data...")
short_term_embeddings, long_term_embeddings, skipped_users = process_semantic_data(semantic_data)
end_time = time.time()

print(f"Number of skipped users: {len(skipped_users)}")
print(f"Processing completed, time spent: {end_time - start_time:.2f} seconds")

print("Saving embedding vectors...")
torch.save(short_term_embeddings, 'semantic_short.pt')
torch.save(long_term_embeddings, 'semantic_long.pt')

print("Short-term and long-term interest embeddings have been successfully saved")

with open('skipped_users.json', 'w', encoding='utf-8') as f:
    json.dump(skipped_users, f, ensure_ascii=False, indent=4)

print("\nVerifying result format:")
sample_user = list(short_term_embeddings.keys())[0]
print(f"Example user: {sample_user}")
print(f"Number of short-term interests: {len(short_term_embeddings[sample_user])}")
print(f"First short-term interest embedding dimension: {short_term_embeddings[sample_user][0].shape}")
print(f"Long-term interest embedding dimension: {long_term_embeddings[sample_user].shape}")