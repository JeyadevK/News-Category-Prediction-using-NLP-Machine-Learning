# 📰 News Category Prediction using Machine Learning

This project aims to classify news stories into categories based on their textual content using NLP and machine learning.

## 📁 Dataset
- `Data_Train.csv`: Training dataset with columns `STORY` (text) and `SECTION` (target category).
- `Data_Test.csv`: Test dataset with only the `STORY` column.
- Output predictions are saved to `outputs/predictions.csv`.

## 🛠️ Technologies Used
- Python
- Pandas, Scikit-learn, Matplotlib
- TF-IDF Vectorization
- Logistic Regression Classifier

## 📈 Visual Outputs
- `outputs/label_distribution.png`: Distribution of classes in training data.
- `outputs/confusion_matrix.png`: Model confusion matrix on validation set.

## 📂 Files
- `Predict_The_News_Category.py`: Complete training, validation, and prediction pipeline.
- `Predict_The_News_Category.ipynb`: Interactive notebook version.
- `requirements.txt`: Install dependencies via `pip install -r requirements.txt`.

## 🚀 Getting Started

### 1. Clone this repository
```bash
git clone https://github.com/JeyadevK/News-Category-Prediction-using-NLP-Machine-Learning
cd News-Category-Prediction-using-NLP-Machine-Learning

**### 2. Run the script**
```bash
python Predict_The_News_Category.py

**### 3. Output**
Predictions will be saved in outputs/predictions.csv.
