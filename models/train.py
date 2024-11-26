import torch
from tqdm import tqdm


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, gradient_clipping_value, log_interval, save_path):
    """
    :param model:
    :param train_loader:
    :param val_loader:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param device:
    :param num_epochs:
    :param gradient_clipping_value:
    :param log_interval:
    :param save_path:
    :return:
    """
    for epoch in range(num_epochs):

        total_loss = 0
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc="Training")):
            model.train()
            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, labels)

            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_value)

            optimizer.step()

            if batch_idx % log_interval == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                torch.save(model.state_dict(), save_path)
                validate_model(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader):.4f}")

        validate_model(model, val_loader, criterion, device)


def validate_model(model, val_loader, criterion, device):
    """
    Validation loop for the translation model.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(val_loader, desc="Validating"):

            outputs = model(input_ids, attention_mask, labels)

            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
