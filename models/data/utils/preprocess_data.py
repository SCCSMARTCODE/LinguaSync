import pandas as pd
from build_custom_tokenizer import tokenize_sentence, Tokenizer
import torch
import os

CSV_PATH = "../eng_french.csv"
PROCESSED_PATH = "../tokenized_dataset.pt"

if os.path.exists(CSV_PATH):
    print("This path does exist.")
else:
    raise FileNotFoundError(f"CSV file '{CSV_PATH}' does not exist.")

df = pd.read_csv(CSV_PATH)

english = df['English words/sentences'].tolist()
french = df['French words/sentences'].tolist()

print("Sample English sentences:", english[:5])
print("Sample French sentences:", french[:5])

en_tokenizer = Tokenizer()
fr_tokenizer = Tokenizer()

en_tokenizer.build_vocab(english)
fr_tokenizer.build_vocab(french)

tokenized_data = {
    "input_ids": [],
    "attention_mask": [],
    "labels": []
}
# i = 0
for eng, fr in zip(english, french):
    eng_tokens = tokenize_sentence(en_tokenizer, eng, max_length=64)
    fr_tokens = tokenize_sentence(fr_tokenizer, fr, max_length=64)

    tokenized_data["input_ids"].append(eng_tokens["input_ids"])
    tokenized_data["attention_mask"].append(eng_tokens["attention_mask"])
    tokenized_data["labels"].append(fr_tokens["input_ids"])
    # print(tokenized_data)

tokenized_data = {key: torch.tensor(value) for key, value in tokenized_data.items()}

torch.save(tokenized_data, PROCESSED_PATH)
print(f"Tokenized dataset saved at '{PROCESSED_PATH}' with english of size {en_tokenizer.vocab_size} and french  of size {fr_tokenizer.vocab_size} ")
