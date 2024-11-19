
# **NMT-Sync: Neural Machine Translation System**

## **Overview**
**NMT-Sync** is a state-of-the-art Machine Translation system that leverages advanced NLP techniques to translate text between languages. The project combines a powerful encoder-decoder architecture with an interactive Flask web application for real-time translations. It is optimized for deployment using Docker to ensure scalability and portability.

---

## **Features**
- **Advanced Architecture**:
  - **Encoder**: Pretrained BERT model for extracting contextual embeddings.
  - **Decoder**: Transformer layers for efficient sequence generation.
- **Efficient Tokenization**:
  - Implements Byte Pair Encoding (BBPE) for robust handling of rare and unknown words.
- **Interactive Web Interface**:
  - A Flask-powered app to input text and receive translations in real-time.
- **Optimized Deployment**:
  - Fully containerized using Docker for ease of deployment.

---

## **Project Workflow**

### **1. Dataset**
- **Source**: WMT (Workshop on Machine Translation) datasets.
- **Languages**: Currently supports English ↔ German with plans to expand.
- **Preprocessing**:
  - Tokenization, cleaning, and BBPE encoding of the dataset.

### **2. Model Architecture**
- Combines a BERT encoder with Transformer-based decoding layers for high-quality translations.

### **3. Training**
- **Framework**: PyTorch.
- **Optimizations**:
  - Teacher forcing for faster convergence.
  - Masking `<PAD>` tokens during loss calculation to improve accuracy.

---

## **Web Application**
- **Backend**: Flask REST API for model interaction.
- **Frontend**: User-friendly input-output interface for translations.

---

## **Deployment**
- **Containerization**: Dockerized for scalability and portability.
- **Supported Platforms**: Deployable on AWS, Heroku, and GCP.

---

## **Evaluation**
- **Metrics**:
  - BLEU, ROUGE, and METEOR scores to evaluate model performance.
- **Visualization**:
  - Side-by-side comparison of predictions and reference translations.

---

## **Expected Outcomes**
1. A functional NMT system with high BLEU scores for translation tasks.
2. Flask-based real-time translation interface.
3. A Dockerized system ready for deployment.

---

## **Future Enhancements**
- Expand to support additional language pairs.
- Integrate with advanced Transformer variants like MarianMT or T5.
- Fine-tune the model for domain-specific translations.

---

## **License**
This project is licensed under the MIT License. Refer to the LICENSE file for details.

---
