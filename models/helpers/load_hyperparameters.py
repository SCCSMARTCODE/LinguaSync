import os
from dotenv import load_dotenv

load_dotenv()


class Hyperparameters:
    # Model Parameters
    D_MODEL = int(os.getenv("D_MODEL"))
    DIM_FF = int(os.getenv("DIM_FF"))
    NUM_HEADS = int(os.getenv("NUM_HEADS"))
    NUM_LAYERS = int(os.getenv("NUM_LAYERS"))
    DROPOUT = float(os.getenv("DROPOUT"))
    VOCAB_SIZE = int(os.getenv("VOCAB_SIZE"))

    # Training Parameters
    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY"))
    SCHEDULER_GAMMA = float(os.getenv("SCHEDULER_GAMMA"))
    GRADIENT_CLIPPING = float(os.getenv("GRADIENT_CLIPPING"))

    # Dataset and Paths
    TOKENIZED_DATA_PATH = os.getenv("TOKENIZED_DATA_PATH")
    MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")

    # Logging
    LOG_INTERVAL = int(os.getenv("LOG_INTERVAL"))
