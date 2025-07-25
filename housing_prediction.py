import pandas as pd
import subprocess
subprocess.run(['pip', 'install', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'joblib', '-q'])
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


# 1. DATA LOADING & PREPROCESSING
print("=== 1. DATA LOADING & PREPROCESSING ===\n")


housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target

# Display dataset information
print("Dataset Shape:", df.shape)
print("\nFeature Names:", housing.feature_names)
print("\nTarget: Median house value in hundreds of thousands of dollars")
print("\nDataset Description:")
print(housing.DESCR[:500] + "...")

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# 2. EXPLORATORY DATA ANALYSIS
print("\n=== 2. EXPLORATORY DATA ANALYSIS ===\n")

# Create visualizations
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

# Plot histograms for all features
for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=30, edgecolor='black')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()

# Pairplot for selected features
selected_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'target']
sns.pairplot(df[selected_features].sample(1000), diag_kind='kde')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()

# 3. DATA PREPROCESSING
print("\n=== 3. DATA PREPROCESSING ===\n")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. MODEL BUILDING
print("\n=== 4. MODEL BUILDING ===\n")

# Linear Regression Model
print("Training Linear Regression Model...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test_scaled)

# Evaluate Linear Regression
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

print("\nLinear Regression Performance:")
print(f"MAE: ${lr_mae*100000:.2f}")
print(f"MSE: {lr_mse:.4f}")
print(f"R² Score: {lr_r2:.4f}")

print("\nTraining Random Forest Model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Making predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Performance:")
print(f"MAE: ${rf_mae*100000:.2f}")
print(f"MSE: {rf_mse:.4f}")
print(f"R² Score: {rf_r2:.4f}")

# 5. HYPERPARAMETER TUNING (BONUS)
print("\n=== 5. HYPERPARAMETER TUNING ===\n")

# Grid Search for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("Performing Grid Search for Random Forest...")
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate best model
best_rf_model = grid_search.best_estimator_
best_rf_predictions = best_rf_model.predict(X_test)
best_rf_r2 = r2_score(y_test, best_rf_predictions)
print(f"Best model R² on test set: {best_rf_r2:.4f}")

# 6. FEATURE IMPORTANCE (BONUS)
print("\n=== 6. FEATURE IMPORTANCE ===\n")

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# 7. MODEL COMPARISON VISUALIZATION
print("\n=== 7. MODEL COMPARISON ===\n")

# Predictions vs Actual plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Linear Regression
ax1.scatter(y_test, lr_predictions, alpha=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Price')
ax1.set_ylabel('Predicted Price')
ax1.set_title(f'Linear Regression\nR² = {lr_r2:.4f}')

# Random Forest
ax2.scatter(y_test, best_rf_predictions, alpha=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Price')
ax2.set_ylabel('Predicted Price')
ax2.set_title(f'Random Forest (Tuned)\nR² = {best_rf_r2:.4f}')

plt.tight_layout()
plt.show()

# 8. SAVE MODELS AND SCALER
print("\n=== 8. SAVING MODELS ===\n")

# Save the best model and scaler
joblib.dump(best_rf_model, 'best_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Models saved successfully!")

# Model comparison summary
print("\n=== MODEL COMPARISON SUMMARY ===")
comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Random Forest (Tuned)'],
    'MAE': [lr_mae, rf_mae, mean_absolute_error(y_test, best_rf_predictions)],
    'MSE': [lr_mse, rf_mse, mean_squared_error(y_test, best_rf_predictions)],
    'R² Score': [lr_r2, rf_r2, best_rf_r2]
})
print(comparison_df)

# Create a requirements.txt file content
requirements_content = """streamlit==1.28.1
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
joblib==1.3.0
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements_content)
print("\nrequirements.txt created successfully!")
