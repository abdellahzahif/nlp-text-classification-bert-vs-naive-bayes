# NLP Text Classification: Naive Bayes vs. BERT (NYTimes Dataset) 📊🤖

> Developed at Hochschule RheinMain | Summer Semester 2026  
> Author: **Abdellah Zahif**

---

## 📌 Project Overview
This project implements a complete Natural Language Processing (NLP) text classification pipeline to categorize New York Times articles into 8 distinct categories (e.g., *arts, business, sports, technology, dining, travel, health, world*). It contrasts a classical statistical approach (**Bernoulli Naive Bayes**, built from scratch) with a modern Deep Learning approach (**Fine-Tuned BERT Transformer**).

---

## 🏗️ System Architecture & Modules

### Part 1: Baseline Model – Bernoulli Naive Bayes (`classifier.py`)
* **Feature Extraction:** Boolean Bag-of-Words features where term frequency within a document is treated as binary ($1$ or $0$).
* **Smoothing & Stability:** Implements **Epsilon-Smoothing** to handle zero-frequency issues and utilizes logarithmic probabilities ($\log P(x_i|c)$) to prevent numerical underflows.
* **Optimization & Persistence:** Evaluated via Grid Search for optimal $\epsilon$ values (achieving ~81.20% accuracy with $\epsilon = 10^{-3}$) and serialized using `pickle` (`model.pkl`).

### Part 2: Advanced Deep Learning – BERT Transformer (`nytimes_bert/`)
* **Framework:** Built using **PyTorch**, **PyTorch Lightning**, and Hugging Face `transformers` (`bert-base-cased`).
* **Architecture:** Extracts sequence embeddings, feeding the primary `[CLS]` token representation into a fully connected linear layer mapped to the 8 output classes.
* **Experiment Tracking:** Integrated with **Weights & Biases (W&B)** for real-time loss and training metric monitoring across epochs.

---

## 🚀 Tech Stack
* **Language:** Python
* **Machine Learning & NLP:** PyTorch, PyTorch Lightning, Transformers (Hugging Face), NumPy, Scikit-Learn
* **Experiment Tracking:** Weights & Biases (W&B)

---

## ⚙️ How to Run

### 1. Run Naive Bayes Classifier

```bash
# Train the model
python classifier.py --train
```

```bash
# Apply/Test the model and evaluate accuracy
python classifier.py --apply
```

### 2. Run BERT Transformer Training & Testing

```bash
# Set up virtual environment
python -m venv myenv
```
```bash
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```
```bash
pip install -r requirements.txt
```

```bash
# Train the BERT model (tracks progress via W&B)
python train.py data/train.json data/test.json
```

```bash
# Evaluate the trained checkpoint
python test.py checkpoints/2.cpt data/test.json
```















