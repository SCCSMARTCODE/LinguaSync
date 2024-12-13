# **NMT-Sync: Neural Machine Translation System**

## **Overview**
**NMT-Sync** is a custom-built Neural Machine Translation (NMT) system, now focusing on English ↔ French translations. It implements a Transformer model from scratch, uses Hugging Face tools for tokenization and dataset management, and provides a user-friendly web interface powered by Flask. The system is fully containerized with Docker to ensure portability and scalability.

---

## **Features**
- **Custom Transformer Model**:
  - Fully implemented from scratch with modular components for extensibility.
  - Optimized initialization for stability and faster convergence.
- **Tokenizer Training**:
  - Custom Byte Pair Encoding (BPE) tokenizer using Hugging Face's `tokenizers` library.
- **Dataset Handling**:
  - Automatic loading and preprocessing of English ↔ French datasets from the Hugging Face hub.
- **Interactive Web Application**:
  - Flask-based app for real-time translation between English and French.
- **Scalable Deployment**:
  - Dockerized for cross-platform deployment.

---

## **New Project Workflow**

### **1. Dataset**
- **Source**: Hugging Face Datasets Library.
- **Languages**: English ↔ French.
- **Preprocessing**:
  - Cleaning, tokenization, and splitting into train, validation, and test sets.
  - Scripted pipelines for automated dataset processing.

### **2. Model Architecture**
- **Transformer**:
  - Implements key components such as:
    - Multi-head attention.
    - Feed-forward networks.
    - Positional encodings.
  - Optimized weight initialization using techniques like Xavier or Kaiming.
- Modular design to separate encoder, decoder, and attention mechanisms for flexibility.

### **3. Tokenizer**
- **Custom Tokenizer**:
  - Trained using Hugging Face `tokenizers` library with BPE.
  - Generates a vocabulary file for seamless integration into the pipeline.

### **4. Training**
- **Framework**: PyTorch.
- **Optimizations**:
  - Scheduled learning rate (e.g., warmup decay).
  - Masking `<PAD>` tokens during loss calculation.
  - Gradient clipping to handle exploding gradients.
- **Loss Function**:
  - Cross-entropy loss with attention masking.

---

## **Web Application**
- **Backend**:
  - Flask REST API for interaction with the model's inference engine.
- **Frontend**:
  - Minimalist interface for entering text and displaying translations.
- **Key Features**:
  - Displays both input and output translations.
  - Logs translation history for debugging and evaluation.

---

## **Deployment**
- **Containerization**:
  - Fully containerized with a `Dockerfile` and `docker-compose.yml` for easy deployment.
- **Platforms**:
  - Supports AWS, Heroku, and Google Cloud for seamless deployment.
- **Environment Variables**:
  - `.env` file for managing configurations and secrets.

---

## **Evaluation**
- **Metrics**:
  - BLEU, ROUGE, and METEOR scores to assess translation accuracy.
- **Visualization**:
  - Attention heatmaps for analyzing word alignments in translations.

---

## **Project Structure**
```
NMT-Sync/
│
├── src/
│   ├── model/
│   │   ├── encoder/               # Transformer encoder implementation
│   │   ├── decoder/               # Transformer decoder implementation
│   │   ├── attention/             # Multi-head attention mechanisms
│   │   ├── layers/                # Core layers (feed-forward, normalization, etc.)
│   │   ├── utils/                 # Utility functions for modeling
│   │   └── __init__.py            # Initialization script for the model package
│   │
│   ├── tokenizer/
│   │   ├── train_tokenizer.py     # Script for training Hugging Face tokenizer
│   │   └── vocab.json             # Vocabulary file generated by tokenizer
│   │
│   ├── dataset/
│   │   ├── data_loader.py         # Dynamic data loading and preprocessing
│   │   └── huggingface_dataset.py # Script for fetching datasets from Hugging Face
│   │
│   ├── training/
│   │   ├── train.py               # Model training script
│   │   ├── scheduler.py           # Learning rate scheduler
│   │   └── loss.py                # Loss function implementations
│   │
│   ├── evaluation/
│   │   ├── metrics.py             # BLEU, ROUGE, METEOR metrics implementation
│   │   └── visualize.py           # Visualization tools (e.g., attention heatmaps)
│   │
│   ├── flask_app/
│   │   ├── app.py                 # Main Flask application
│   │   ├── templates/             # HTML templates
│   │   └── static/                # CSS, JS, and assets for the Flask app
│   │
│   └── inference/
│       ├── translate.py           # Model inference logic
│       └── batch_translate.py     # Batch translation script
│
├── tests/
│   ├── test_encoder.py            # Unit tests for encoder
│   ├── test_decoder.py            # Unit tests for decoder
│   ├── test_tokenizer.py          # Unit tests for tokenizer
│   └── test_translation.py        # Integration tests
│
├── benchmarks/
│   ├── performance.py             # Performance benchmarks
│   └── comparison.py              # Model comparison scripts
│
├── docs/
│   ├── architecture.md            # Details on the Transformer architecture
│   ├── usage.md                   # Guide to using the NMT system
│   └── web_app.md                 # Documentation for the Flask app
│
├── .env                           # Environment configuration file
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker build configuration
├── docker-compose.yml             # Multi-container orchestration
├── LICENSE                        # License file
└── README.md                      # Project overview
```

---

## **Expected Outcomes**
1. A custom NMT system capable of high-quality English ↔ French translations.
2. Fully containerized and deployable Flask web application.
3. Comprehensive evaluation metrics and visualization tools.

---

## **Future Enhancements**
- **Additional Languages**: Expand support for other language pairs.
- **Advanced Architectures**: Experiment with pre-trained models like T5 or MarianMT.
- **Domain-Specific Training**: Fine-tune the model for specific industries (e.g., legal, medical).
- **Improved Web Interface**: Add features such as batch translation and file uploads.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
