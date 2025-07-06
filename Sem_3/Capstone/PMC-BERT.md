# 📘 Clinical Treatment Summary Generator using BERT Encoder + Transformer Decoder
# Dataset: Single-column text corpus (~160k records)
# Objective: Generate summaries of treatments from clinical case descriptions using pretrained BERT embeddings

```py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import random

# ✅ Step 1: Load Dataset

df = pd.read_csv("your_dataset.csv")
texts = df["text"].astype(str).tolist()

# ✅ Step 2: Tokenization and Embedding using BERT

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# Freeze BERT to avoid training
for param in bert_model.parameters():
    param.requires_grad = False

# ✅ Step 3: Custom Dataset and Collator

class ClinicalDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_len)
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    return input_ids, attention_masks

# ✅ Step 4: Decoder Model (Transformer)

class DecoderTransformer(nn.Module):
    def __init__(self, d_model=768, vocab_size=30522, num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model))
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        seq_len = tgt.size(1)
        x = self.embedding(tgt) + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x.transpose(0, 1), memory.transpose(0, 1))
        return self.fc_out(x.transpose(0, 1))

# ✅ Step 5: Full Model Wrapper

class TreatmentSummaryModel(nn.Module):
    def __init__(self, bert, decoder):
        super().__init__()
        self.bert = bert
        self.decoder = decoder

    def forward(self, input_ids, attention_mask, tgt):
        with torch.no_grad():
            encoder_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return self.decoder(tgt, encoder_outputs)

# ✅ Step 6: Training Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder = DecoderTransformer().to(device)
model = TreatmentSummaryModel(bert_model, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Dummy decoder input (in practice, train with targets if available)
tgt_start = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Treatment summary:"))
tgt_start_tensor = torch.tensor([tgt_start], dtype=torch.long).to(device)

# ✅ Step 7: Train on Single Epoch (example)
dataset = ClinicalDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

model.train()
for epoch in range(1):
    total_loss = 0
    for input_ids, attention_mask in dataloader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        tgt = tgt_start_tensor.repeat(input_ids.size(0), 1)

        output = model(input_ids, attention_mask, tgt)
        loss = criterion(output.view(-1, output.shape[-1]), tgt.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# ✅ Step 8: Inference

def generate_summary(text, max_len=100):
    model.eval()
    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        input_ids, attention_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        memory = bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]

        generated = tgt_start_tensor
        for _ in range(max_len):
            out = decoder(generated, memory)
            next_token = out[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.sep_token_id:
                break
        return tokenizer.decode(generated[0], skip_special_tokens=True)

# ✅ Example Output
print(generate_summary("This 60-year-old male was hospitalized due to moderate ARDS from COVID-19 with symptoms of fever..."))
```