import torch
from torch.utils.data import random_split, DataLoader
from data.utils.dataset_utils import TranslationDataset, CustomDL
from models.helpers.load_hyperparameters import Hyperparameters
from models.data.utils.preprocess_data import fr_tokenizer, en_tokenizer
from models.helpers.performance_utils import cross_entropy_loss, get_optimizer, get_scheduler, gradient_clipping
from models.model_architecture import NMTModel
from models.train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("Loading hyperparameters...")
    hparams = Hyperparameters()

    print("Loading dataset...")
    dataset = TranslationDataset(tokenized_data_path=hparams.TOKENIZED_DATA_PATH)

    train_size = int(0.9 * len(dataset))
    val_size = int(0.1 * (len(dataset) - train_size))
    test_size = int(len(dataset)-train_size-val_size)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=hparams.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams.BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=hparams.BATCH_SIZE, shuffle=False, drop_last=True)

    train_loader = CustomDL(train_loader)
    val_loader = CustomDL(val_loader)
    test_loader = CustomDL(test_loader)

    print("Initializing model...")

    model = NMTModel(
        en_vocab_size=hparams.ENGLISH_VOCAB_SIZE,
        fr_vocab_size=hparams.FRENCH_VOCAB_SIZE,
        embed_size=hparams.D_MODEL,
        hidden_size=hparams.D_MODEL,
        num_layers=hparams.NUM_LAYERS,
        num_heads=hparams.NUM_HEADS,
        dropout=hparams.DROPOUT
    )
    model = model.to(device)

    print("Setting up loss, optimizer, and scheduler...")
    criterion = cross_entropy_loss(ignore_index=fr_tokenizer.token_to_idx["<pad>"])  # PAD token is indexed as 3
    optimizer = get_optimizer(model, hparams.LEARNING_RATE, hparams.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, hparams.SCHEDULER_GAMMA)

    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=hparams.NUM_EPOCHS,
        gradient_clipping_value=hparams.GRADIENT_CLIPPING,
        log_interval=hparams.LOG_INTERVAL,
        save_path=hparams.MODEL_SAVE_PATH
        )


if __name__ == '__main__':
    main()
