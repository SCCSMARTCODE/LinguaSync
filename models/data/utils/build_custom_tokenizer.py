import os
import pandas as pd
import re
from collections import Counter


class Tokenizer:
    def __init__(self, special_tokens=None):

        """
        Initialize the Tokenizer with optional special tokens.

        :param special_tokens: List of special tokens like <pad>, <unk>, <s>, </s>.
        """
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]

        self.special_tokens = special_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.is_vocab_created = False

    def build_vocab(self, sentences, min_frequency=2):
        """
        Build a vocabulary from a list of sentences.

        :param sentences: List of strings (sentences) to build the vocabulary from.
        :param min_frequency: Minimum frequency for a word to be included in the vocabulary.
        """
        # Clean and tokenize sentences
        words = []
        for sentence in sentences:
            words.extend(self._clean_and_tokenize(sentence))

        # Count word frequencies
        word_counts = Counter(words)

        # Add words meeting the min_frequency criteria
        for word, count in word_counts.items():
            if count >= min_frequency and word not in self.token_to_idx:
                idx = len(self.token_to_idx)
                self.token_to_idx[word] = idx
                self.idx_to_token[idx] = word
        self.is_vocab_created = True

    def tokenize(self, sentence):
        """
        Tokenize a sentence into a list of token indices.

        :param sentence: Input sentence as a string.
        :return: List of token indices.
        """
        if not self.is_vocab_created:
            print("Vocab is not available...")
        tokens = self._clean_and_tokenize(sentence)
        return [self.token_to_idx.get(token, self.token_to_idx["<unk>"]) for token in tokens]

    def detokenize(self, token_indices):
        if not self.is_vocab_created:
            print("Vocab is not available...")
        """
        Convert a list of token indices back to a sentence.

        :param token_indices: List of token indices.
        :return: Detokenized sentence as a string.
        """
        tokens = [self.idx_to_token.get(idx, "<unk>") for idx in token_indices]
        return " ".join(tokens)

    @property
    def vocab_size(self):
        if not self.is_vocab_created:
            print("Vocab is not available...")
        """Return the size of the vocabulary."""
        return len(self.token_to_idx)

    def _clean_and_tokenize(self, sentence):
        """
        Clean and tokenize a sentence.

        :param sentence: Input sentence as a string.
        :return: List of cleaned tokens.
        """
        sentence = re.sub(r"[^\w\s]", "", sentence.lower())
        return sentence.split()


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


def tokenize_sentence(tokenizer: Tokenizer, sentence: str, max_length: int):
    """
    Tokenizes a given sentence using the provided tokenizer.

    :param tokenizer: Instance of the Tokenizer class.
    :param sentence: Sentence to tokenize.
    :param max_length: Maximum length for padding/truncation.
    :return: Dictionary containing input IDs and attention mask.
    """
    token_ids = tokenizer.tokenize(sentence)

    special_tokens = {
        "<s>": tokenizer.token_to_idx["<s>"],
        "</s>": tokenizer.token_to_idx["</s>"],
        "<pad>": tokenizer.token_to_idx["<pad>"]
    }

    token_ids = [special_tokens["<s>"]] + token_ids[:max_length - 2] + [special_tokens["</s>"]]

    padded_ids = token_ids + [special_tokens["<pad>"]] * (max_length - len(token_ids))

    attention_mask = [1] * len(token_ids) + [0] * (max_length - len(token_ids))

    padded_ids = padded_ids[:max_length]
    attention_mask = attention_mask[:max_length]

    return {
        "input_ids": padded_ids,
        "attention_mask": attention_mask
    }
