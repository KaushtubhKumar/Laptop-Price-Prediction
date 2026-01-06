# Laptop Price Prediction 💻📊

A machine learning project that predicts laptop prices based on hardware specifications and features. This model was trained on real laptop data and uses preprocessing, feature engineering, and regression techniques for fairly accurate predictions.

## 🚀 Project Overview

Modern laptops come with a vast array of configurations. This project demonstrates how to build an ML model that takes those specs and predicts the expected price. It includes data exploration, model training, and a prediction pipeline for real-world use.

## 📁 Repository Structure

| File / Folder | Description |
|---------------|-------------|
| `laptop_predictor.ipynb` | Jupyter notebook with complete data preprocessing, EDA, modeling, and evaluation. |
| `main.py` | Python script to run the prediction pipeline (or app entry point). |
| `laptop_data.csv` | Raw dataset with laptop specifications and prices. |
| `df.pkl`, `pipe.pkl` | Pickled processed data and trained model pipeline. |
| `requirements.txt` | Required Python libraries. |
| `.gitignore` | List of files/folders ignored by Git. |
| `Procfile` | Deployment config (for platforms like Heroku). |

## 🛠️ Features Used

The model considers specs such as:
- Brand / Company  
- Type of laptop (Notebook, Ultrabook, etc.)  
- Screen size & resolution  
- CPU brand and speed  
- RAM, Storage type & size  
- GPU  
- Operating System  
- Weight  

These features are processed and encoded before feeding them to the regression model.

## 🧠 How It Works

1. **Load data**  
2. **Clean and preprocess** (handle missing values, parse text features)  
3. **Feature engineering** (e.g., splitting screen resolution, CPU model)  
4. **Train ML model** (regressors like Random Forest / others)  
5. **Save pipeline and use it for predictions**

A similar pipeline approach is used in other laptop price prediction repos online. :contentReference[oaicite:0]{index=0}

## 📦 Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/KaushtubhKumar/Laptop-Price-Prediction.git
   cd Laptop-Price-Prediction
