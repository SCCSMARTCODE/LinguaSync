import pandas as pd
# from transformers import BertTokenizerFast
from tokenizers import ByteLevelBPETokenizer
from build_custom_tokenizer import tokenize_sentence
import torch
import os

CSV_PATH = "../eng_french.csv"
PROCESSED_PATH = "../tokenized_dataset.pt"

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename="tokenizer/vocab.json",
    merges_filename="tokenizer/merges.txt"
)
print(f"My Tokenizer has vocab size of [ {tokenizer.get_vocab_size()} ]")


if os.path.exists(CSV_PATH):
    print("This path does Exists")
else:
    print("This path does not Exist")
df = pd.read_csv(CSV_PATH)

english = df['English words/sentences'].tolist()
french = df['French words/sentences'].tolist()

print(english[:20])
print(french[:20])

tokenized_data = {
    "input_ids": [],
    "attention_mask": [],
    "labels": []
}

for eng, fr in zip(english, french):
    eng_tokens = tokenize_sentence(tokenizer, eng, max_length=128)

    fr_tokens = tokenize_sentence(tokenizer, fr, max_length=128)

    # print(eng, eng_tokens['input_ids'], eng_tokens['attention_mask'], sep='\n')
    # print(fr, fr_tokens['input_ids'], fr_tokens['attention_mask'], sep='\n')

    # Add to dataset
    tokenized_data["input_ids"].append(eng_tokens["input_ids"])
    tokenized_data["attention_mask"].append(eng_tokens["attention_mask"])
    tokenized_data["labels"].append(fr_tokens["input_ids"])

tokenized_data = {key: torch.tensor(value) for key, value in tokenized_data.items()}

torch.save(tokenized_data, PROCESSED_PATH)
