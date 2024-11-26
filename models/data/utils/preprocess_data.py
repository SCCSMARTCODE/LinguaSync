import pandas as pd
from transformers import BertTokenizerFast
import torch
import os

CSV_PATH = "../english_french_sample.csv"
PROCESSED_PATH = "../tokenized_dataset.pt"

tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
print(f"My Tokenizer has vocab size of [ {tokenizer.vocab_size} ]")

if os.path.exists(CSV_PATH):
    print("This path does Exists")
else:
    print("This path does not Exist")
df = pd.read_csv(CSV_PATH)

english = df['English'].tolist()
french = df['French'].tolist()

print(english[:20])
print(french[:20])

tokenized_data = {
    "input_ids": [],
    "attention_mask": [],
    "labels": []
}

for eng, fr in zip(english, french):
    eng_tokens = tokenizer(eng, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    fr_tokens = tokenizer(fr, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    # Add to dataset
    tokenized_data["input_ids"].append(eng_tokens["input_ids"].squeeze(0))
    tokenized_data["attention_mask"].append(eng_tokens["attention_mask"].squeeze(0))
    tokenized_data["labels"].append(fr_tokens["input_ids"].squeeze(0))

tokenized_data = {key: torch.stack(value) for key, value in tokenized_data.items()}

torch.save(tokenized_data, PROCESSED_PATH)
