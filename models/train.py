import torch
from tqdm import tqdm
from data.utils.build_custom_tokenizer import tokenize_sentence
from data.utils.preprocess_data import fr_tokenizer, en_tokenizer


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

            # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_value)

            optimizer.step()

            if batch_idx % log_interval == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                torch.save(model.state_dict(), save_path)
                evaluate_translation(model, en_tokenizer, fr_tokenizer, 'Good morning Guys', device)
            if batch_idx % (log_interval*2) == 0:
                val_loss = validate_model(model, val_loader, criterion, device)

                # wandb.log({
                #     "cur-train-loss": loss.item(),
                #     "cur-val-loss": val_loss
                # })

        print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader):.4f}")

        validate_model(model, val_loader, criterion, device)

        scheduler.step()
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
        for input_ids, attention_mask, labels in val_loader:

            outputs = model(input_ids, attention_mask, labels)

            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_translation(model, en_tokenizer, fr_tokenizer, input_text, device, max_len=64):
    """
    Generate a translation for the given input text using the custom tokenizer and model.

    :param model: Trained NMT model.
    :param fr_tokenizer: Custom tokenizer instance with decode functionality
    :param en_tokenizer: Custom tokenizer instance with decode functionality
    :param input_text: Text in the source language (English).
    :param device: Device (CPU or GPU).
    :param max_len: Maximum length for generated sequences.
    """
    model.eval()
    with torch.no_grad():
        # Tokenize the input text
        tokenized = tokenize_sentence(en_tokenizer, input_text, max_length=max_len)
        input_ids = torch.tensor([tokenized['input_ids']], dtype=torch.long).to(device)
        attention_mask = torch.tensor([tokenized['attention_mask']], dtype=torch.long).to(device)

        # Retrieve special token IDs
        bos_token_id = fr_tokenizer.token_to_idx["<s>"]
        eos_token_id = fr_tokenizer.token_to_idx["</s>"]
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

            # Add the predicted token to the input for the next time step
            tgt_input_ids = torch.cat([tgt_input_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(device)], dim=1)

        # Detokenize the generated IDs to get the translated text
        generated_text = fr_tokenizer.detokenize(generated_ids)

        # Display the input and translated output
        print(f"Input (English): {input_text}")
        print(f"Output (French): {generated_text}")
