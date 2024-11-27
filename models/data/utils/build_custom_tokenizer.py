from tokenizers import ByteLevelBPETokenizer
import os
import pandas as pd


def train_tokenizer(data_file, output_dir, vocab_size=30000, min_frequency=2):
    """
    Train a Byte-Pair Encoding (BBPE) tokenizer on a given text file.

    :param data_file: Path to the dataset text file.
    :param output_dir: Directory where the tokenizer files will be saved.
    :param vocab_size: Size of the vocabulary.
    :param min_frequency: Minimum frequency of a token to be included in the vocabulary.
    """
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[data_file],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>"]
    )

    # Save the trained tokenizer
    tokenizer.save_model(output_dir)
    print(f"Tokenizer trained and saved at {output_dir}")


def make_txt(csv_path, dest_path):
    if os.path.exists(dest_path):
        print("File already exist")
        return
    if os.path.exists(csv_path):
        file = pd.read_csv(csv_path)

        with open(dest_path, 'w', encoding="UTF-8") as f:
            for eng, fr in zip(file['English words/sentences'].tolist(), file['French words/sentences'].tolist()):
                f.write(eng+"\n")
                f.write(fr+"\n")


def tokenize_sentence(tokenizer, sentence, max_length):
    """
    Tokenizes a given sentence using the provided tokenizer.

    :param tokenizer: Trained tokenizer instance.
    :param sentence: Sentence to tokenize.
    :param max_length: Maximum length for padding/truncation.
    :return: Dictionary containing input IDs and attention mask.
    """
    token = tokenizer.encode(sentence)

    token_ids = [tokenizer.token_to_id("<s>")] + token.ids[:max_length - 2] + [tokenizer.token_to_id("</s>")]

    padded_ids = token_ids + [tokenizer.token_to_id("<pad>")] * (max_length - len(token_ids))

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
    make_txt(CSV_FILE, DATA_FILE)
    train_tokenizer(DATA_FILE, OUTPUT_DIR)
