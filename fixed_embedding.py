from utils import load_data, textDataset
from model import BertClassifierInf
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import torch
import os
from utils import init_seed

model_name = 'bert-small'
dataset = 'citeseer'
data = torch.load(f'data/{dataset}/{dataset}_fixed_sbert.pt', map_location='cpu')
print(data)
text = data.raw_texts
print(text[0])
# Example list of texts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrain_model = AutoModel.from_pretrained(f"model/pre_train/{model_name}/")
tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/bert-base/")
X = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
text_dataset = textDataset(X)
text_dataloader = DataLoader(text_dataset, batch_size= 64, shuffle=False)
model = BertClassifierInf(pretrain_model).to(device)

embeddings = []
for batch in text_dataloader:
    with torch.no_grad():  # Inference mode, no gradients needed
        batch_embeddings = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))  # Your model generates embeddings
        embeddings.append(batch_embeddings)

# Concatenate all embeddings (if they are torch tensors)
all_embeddings = torch.cat(embeddings, dim=0)
print(all_embeddings.size())
save_path = f'data/{dataset}/{dataset}_{model_name}.pt'
torch.save(all_embeddings, save_path)
