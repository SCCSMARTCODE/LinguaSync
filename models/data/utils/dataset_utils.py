import torch
from torch.utils.data import Dataset
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


@contextmanager
def custom_dl(data_loader: DataLoader, device):
    """
    A context manager for safely handling data loading and GPU cleanup.

    Args:
        data_loader: PyTorch DataLoader object.
        device: The device (CPU or GPU) to which the data should be moved.

    Yields:
        in_seq, attention_mask, out_seq: Batched data moved to the specified device.
    """
    try:
        for batch in data_loader:
            input_ids, attention_mask, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            yield input_ids, attention_mask, labels

            del input_ids, attention_mask, labels
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Exception occurred during data loading: {e}")
    finally:
        torch.cuda.empty_cache()
