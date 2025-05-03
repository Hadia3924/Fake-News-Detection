# Fake News Detection with Transformers

This project uses transformer-based models (BERT, BART, and a custom transformer) to classify news articles as **Real** or **Fake**. It leverages deep learning and NLP techniques to improve detection accuracy based on article headlines and full text.

## Dataset

- Source: Kaggle / Manually uploaded to Google Drive
- Files:
  - `True.csv`: Contains real news articles
  - `Fake.csv`: Contains fake news articles
- Columns: `title`, `text`, `subject`, `date`

## Models Used

1. **BERT (pretrained)** – fine-tuned on the dataset
2. **BART (pretrained)** – fine-tuned for classification
3. **Custom Transformer** – trained from scratch on the dataset

## Features

- Data Preprocessing and Cleaning
- Tokenization using Hugging Face Transformers
- Training and Validation Splits
- Evaluation using Accuracy and Classification Report
- GPU-compatible training in Google Colab

## How to Run

1. Open the Colab notebook: [DLP_Project(21k3924,21k3834).ipynb](./DLP_Project(21k3924,21k3834).ipynb)
2. Mount Google Drive and load the dataset
3. Run all cells in order
4. Optionally switch runtime to GPU for faster training

## Requirements

- Python 3.8+
- `transformers`, `torch`, `pandas`, `scikit-learn`, `datasets`

Install via pip:

```bash
pip install transformers torch pandas scikit-learn datasets 
