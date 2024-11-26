import torch
from torch.utils.data import Dataset
import gc
from contextlib import contextmanager
from torch.utils.data import DataLoader


class TranslationDataset(Dataset):
    """
        Args:
            tokenized_data_path (str): Path to the preprocessed tokenized dataset (.pt file).
    """
    def __init__(self, tokenized_data_path):
        self.data = torch.load(tokenized_data_path)

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        """
            Returns:
                input_ids (torch.Tensor): Tokenized source sentence (English).
                attention_mask (torch.Tensor): Attention mask for the source sentence.
                labels (torch.Tensor): Tokenized target sentence (French).
        """
        input_ids = self.data["input_ids"][idx]
        attention_mask = self.data["attention_mask"][idx]
        labels = self.data["labels"][idx]

        return input_ids, attention_mask, labels


class CustomDL:
    def __init__(self, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        try:
            for batch in self.dataloader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = (
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                    labels.to(self.device),
                )
                yield input_ids, attention_mask, labels
                del input_ids, attention_mask, labels
                torch.cuda.empty_cache() if self.device == 'cuda' else None
                gc.collect()
        except Exception as e:
            print(f"Exception [ {e} ] occurred when trying to load data to {self.device}")

    def __len__(self):
        return len(self.dataloader)
