# Machine Failure Prediction using NLP & BERT

This project is the final group assignment for the **"Python for Data Science and AI"** course.  
We built a machine learning pipeline using **BERT-based NLP models** to predict machine failures from textual logs.

---

## Project Overview

Industrial machines often generate text-based logs describing operating conditions, warnings, and alerts.  
The goal of this project is to use these logs to **predict machine failures**, enabling early detection and preventive maintenance.

We apply **Natural Language Processing (NLP)** and **BERT (Bidirectional Encoder Representations from Transformers)** for binary classification.

---

## Project Structure

```
📁 Project-Machine-Failure-Prediction-using-NLP-BERT
 ├── Final_project.ipynb          # Full notebook (training, evaluation)
 ├── machine_failure.csv          # Dataset
 └── README.md                    # Documentation
```

---

## Approach

### **1. Data Preprocessing**
- Clean and normalize text
- Handle missing values
- Encode labels (0 = no failure, 1 = failure)

### **2. Tokenization**
- Use `bert-base-uncased` tokenizer
- Convert text to input IDs + attention masks

### **3. Model**
We fine-tuned a **BERT model for sequence classification**:
- Pretrained model: `bert-base-uncased`
- Added dense classification layer
- Trained using:
  - AdamW optimizer  
  - Learning rate scheduler  
  - CrossEntropyLoss  

### **4. Evaluation**
Metrics used:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix  

---

## Results

| Metric | Score |
|--------|--------|
| Accuracy | **XX%** |
| F1-score | **XX%** |
| Precision | **XX%** |
| Recall | **XX%** |

*(Update with your actual scores.)*

---

## Dataset

File: `machine_failure.csv`  
Contains:
- Text logs of machine operations  
- Labels indicating failure state  

Example fields (cập nhật theo dataset thật):
- `log_text`
- `failure`

Dataset statistics:
- Number of samples: X  
- Number of failure cases: X  
- Number of normal cases: X  

---

## How to Run

### ✔ Option 1 — Run on Google Colab
1. Open the notebook  
2. Upload dataset when asked  
3. Run all cells  

### ✔ Option 2 — Run locally

Install dependencies:

```bash
pip install transformers torch pandas numpy scikit-learn
```

Run the notebook via Jupyter or VSCode.

---

## Technologies Used

- Python  
- Hugging Face Transformers  
- PyTorch  
- Scikit-learn  
- Pandas / NumPy  
- Google Colab  

---

## Team Members

— Hoang Khai Minh  
- Mai Anh Nghia  
---

## Notes

- This project focuses on **text-based prediction only**  
- BERT chosen for its strong contextual understanding  
- Future improvements:
  - Try RoBERTa, DistilBERT  
  - Hyperparameter tuning  
  - Text augmentation  

---

## License

This project is for educational purposes only.
