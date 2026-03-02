# 🛡️ YouTube Spam Classifier

This project achieves **~90% accuracy** using an NLP pipeline and **XGBoost**.

## 📊 Dataset Overview
The model merges five separate comment datasets to create a bigger training set:
- **Total Samples:** Aggregated from `d1.csv` through `d5.csv`.
- **Classes:** 
  - `0`: Ham (Real comments)
  - `1`: Spam (Advertisements, scams, links)

## 🛠️ The Pipeline
The code uses a Scikit-Learn `Pipeline` to automate the workflow:
1. **Text Cleaning**: Custom removal of emojis, punctuation, and special characters.
2. **Vectorization**: `CountVectorizer` converts text to word frequencies.
3. **Weighting**: `TfidfTransformer` applies TF-IDF to highlight significant words.
4. **Classification**: `XGBClassifier` (Gradient Boosting) handles the final prediction.

## 🚀 Quick Start

### Installation
```bash
pip install pandas xgboost scikit-learn emoji joblib matplotlib
