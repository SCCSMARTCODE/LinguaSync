import pandas as pd
from tokenizers import Tokenizer
from build_custom_tokenizer import tokenize_sentence
import torch
import os

CSV_PATH = "../eng_french.csv"
PROCESSED_PATH = "../tokenized_dataset.pt"
TOKENIZER_PATH = "tokenizer/bpe_tokenizer.json"

if os.path.exists(CSV_PATH):
    print("This path does exist.")
else:
    raise FileNotFoundError(f"CSV file '{CSV_PATH}' does not exist.")

df = pd.read_csv(CSV_PATH)

english = df['English words/sentences'].tolist()
french = df['French words/sentences'].tolist()

print("Sample English sentences:", english[:5])
print("Sample French sentences:", french[:5])

if os.path.exists(TOKENIZER_PATH):
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"Tokenizer successfully loaded with vocab size: {tokenizer.get_vocab_size()}")
else:
    raise FileNotFoundError(f"Tokenizer file '{TOKENIZER_PATH}' does not exist.")

tokenized_data = {
    "input_ids": [],
    "attention_mask": [],
    "labels": []
}

for eng, fr in zip(english, french):
    eng_tokens = tokenize_sentence(TOKENIZER_PATH, eng, max_length=64)
    fr_tokens = tokenize_sentence(TOKENIZER_PATH, fr, max_length=64)

    tokenized_data["input_ids"].append(eng_tokens["input_ids"])
    tokenized_data["attention_mask"].append(eng_tokens["attention_mask"])
    tokenized_data["labels"].append(fr_tokens["input_ids"])

tokenized_data = {key: torch.tensor(value) for key, value in tokenized_data.items()}

torch.save(tokenized_data, PROCESSED_PATH)
print(f"Tokenized dataset saved at '{PROCESSED_PATH}'")
