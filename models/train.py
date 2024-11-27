import torch
from tqdm import tqdm
from data.utils.build_custom_tokenizer import tokenize_sentence


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


def evaluate_translation(model, tokenizer, input_text, device, max_len=128):
    """
    Generate a translation for the given input text using the custom tokenizer and model.

    :param model: Trained NMT model.
    :param tokenizer: Custom tokenizer with encode/decode functionality.
    :param input_text: Text in the source language (English).
    :param device: Device (CPU or GPU).
    :param max_len: Maximum length for generated sequences.
    """
    model.eval()
    with torch.no_grad():

        tokenized = tokenize_sentence(tokenizer, input_text, max_length=max_len)
        input_ids = torch.tensor([tokenized['input_ids']], dtype=torch.long).to(device)
        attention_mask = torch.tensor([tokenized['attention_mask']], dtype=torch.long).to(device)

        bos_token_id = tokenizer.token_to_id("<s>")
        eos_token_id = tokenizer.token_to_id("</s>")
        tgt_input_ids = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)

        generated_ids = []

        for _ in range(max_len):
            outputs = model(
                src_input_ids=input_ids,
                src_attention_mask=attention_mask,
                tgt_input_ids=tgt_input_ids
            )

            next_token_id = torch.argmax(outputs[:, -1, :], dim=-1).item()
            generated_ids.append(next_token_id)

            if next_token_id == eos_token_id:
                break

            tgt_input_ids = torch.cat([tgt_input_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(device)], dim=1)

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Display the input and translated output
        print(f"Input (English): {input_text}")
        print(f"Output (French): {generated_text}")

