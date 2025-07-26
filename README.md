# California Housing Price Prediction

## Project Summary
This project implements a machine learning solution to predict housing prices in California using the California Housing dataset. The solution includes data preprocessing, exploratory data analysis, model training with hyperparameter tuning, and a user-friendly Streamlit web application for making predictions.

## Features
- **Data Analysis**: Comprehensive EDA with visualizations
- **Models Used**: Linear Regression and Random Forest implementations
- **Hyperparameter Tuning**: GridSearchCV for optimal model performance
- **Criteria for Feature Extraction**: Analysis of critical factors
- **Interactive Web App**: Streamlit interface for user-friendly UI

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Antrita/Predicting-House-Prices-using-Machine-Learning.git
cd california-housing-prediction
```

### 2. Installing Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Jupyter Notebook
```bash
jupyter notebook housing_prediction.ipynb
```
### 4. Launch the Streamlit application
```bash
streamlit run app.py
```
## Model Performance

### Metrics Explanation
- **MAE (Mean Absolute Error)**: Average prediction error in dollars
- **MSE (Mean Squared Error)**: Average squared prediction error
- **R² Score**: Proportion of variance explained (closer to 1 is better)

### Results
| Model | MAE | MSE | R² Score |
|-------|-----|-----|----------|
| Linear Regression | $53,320 | 0.5559 | 0.5758 |
| Random Forest | $32,754 | 0.2554 | 0.8051 |
| Random Forest (Tuned) | $32,681 | 0.2540 | 0.8062 |

## Streamlit App Screenshot
![Streamlit App Interface](Streamlit_UI.png)

## 5. Instructions for Google Colab

To run this in Google Colab:

1. Create a new Colab notebook
2. Copy the Jupyter notebook code into cells
3. Run all cells sequentially
4. Download the generated model files (`best_rf_model.pkl`, `scaler.pkl`)
5. Create `app.py` locally with the Streamlit code
6. Run the Streamlit app locally with the downloaded model files

## Bonus Features Implemented
✅ Feature importance visualization
✅ GridSearchCV hyperparameter tuning
✅ Model saving with joblib
✅ Comprehensive documentation
✅ Interactive Streamlit UI with metrics display

