import os
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers


def train_tokenizer(data_file, output_dir, vocab_size=10000, min_frequency=2):
    """
    Train a BPE tokenizer on a given text file.

    :param data_file: Path to the dataset text file.
    :param output_dir: Directory where the tokenizer files will be saved.
    :param vocab_size: Size of the vocabulary.
    :param min_frequency: Minimum frequency of a token to be included in the vocabulary.
    """
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer(models.BPE())

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    special_tokens = ["<s>", "</s>", "<unk>", "<pad>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens
    )

    tokenizer.train(files=[data_file], trainer=trainer)

    tokenizer.save(os.path.join(output_dir, "bpe_tokenizer.json"))
    print(f"Tokenizer trained and saved at {os.path.join(output_dir, 'bpe_tokenizer.json')}")


def make_txt(csv_path, dest_path):
    """
    Convert a CSV file containing English and French sentences into a plain text file.

    :param csv_path: Path to the CSV file.
    :param dest_path: Path where the text file will be saved.
    """
    if os.path.exists(dest_path):
        print(f"File '{dest_path}' already exists. Skipping text generation.")
        return

    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' does not exist.")
        return

    try:
        # Read CSV file
        file = pd.read_csv(csv_path)

        with open(dest_path, 'w', encoding="UTF-8") as f:
            for eng, fr in zip(file['English words/sentences'], file['French words/sentences']):
                f.write(f"{eng.strip()}\n")
                f.write(f"{fr.strip()}\n")

        print(f"Text file generated at {dest_path}")
    except Exception as e:
        print(f"Error processing CSV file: {e}")


def tokenize_sentence(tokenizer_path, sentence, max_length):
    """
    Tokenizes a given sentence using the provided tokenizer.

    :param tokenizer_path: Path to the trained tokenizer file.
    :param sentence: Sentence to tokenize.
    :param max_length: Maximum length for padding/truncation.
    :return: Dictionary containing input IDs and attention mask.
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file '{tokenizer_path}' not found.")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    token = tokenizer.encode(sentence)
    special_tokens = {"<s>": tokenizer.token_to_id("<s>"), "</s>": tokenizer.token_to_id("</s>"), "<pad>": tokenizer.token_to_id("<pad>")}

    token_ids = [special_tokens["<s>"]] + token.ids[:max_length - 2] + [special_tokens["</s>"]]

    # Padding
    padded_ids = token_ids + [special_tokens["<pad>"]] * (max_length - len(token_ids))

    # Attention mask
    attention_mask = [1] * len(token_ids) + [0] * (max_length - len(token_ids))

    return {
        "input_ids": padded_ids,
        "attention_mask": attention_mask
    }


if __name__ == "__main__":
    # Example usage
    DATA_FILE = "../english_french_corpus.txt"  # Path to your dataset
    OUTPUT_DIR = "tokenizer/"  # Directory to save the tokenizer
    CSV_FILE = "../eng_french.csv"

    # Generate the text file from CSV
    make_txt(CSV_FILE, DATA_FILE)

    # Train the tokenizer
    train_tokenizer(DATA_FILE, OUTPUT_DIR)

    # Example tokenization
    TOKENIZER_PATH = os.path.join(OUTPUT_DIR, "bpe_tokenizer.json")
    example_sentence = "This is a test sentence for tokenization."
    max_len = 20

    try:
        tokenized_output = tokenize_sentence(TOKENIZER_PATH, example_sentence, max_len)
        print("Tokenized Output:", tokenized_output)
    except Exception as e:
        print(f"Error during tokenization: {e}")
