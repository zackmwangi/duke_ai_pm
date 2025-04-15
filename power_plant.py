# %% [markdown]
# # 🔌 Power Plant Energy Output Prediction
# 
# A machine learning project to predict the net hourly electrical energy output (PE) based on environmental variables.
# 
# **Features**:
# - Temperature (T)
# - Exhaust Vacuum (V)
# - Ambient Pressure (AP)
# - Relative Humidity (RH)
# 
# **Target**:
# - Net hourly electrical energy output (PE)

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %%
# Load dataset
df = pd.read_csv('power_plant_data.csv')  # Replace with your actual file name
df.head()

# %% [markdown]
# ## 🔍 Exploratory Data Analysis

# %%
# Basic statistics
df.describe()

# %%
# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# %% [markdown]
# ## ✂️ Data Preprocessing

# %%
# Features and target
X = df[['T', 'V', 'AP', 'RH']]
y = df['PE']

# Train-validation-test split
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# %%
print(f"Train+Val Size: {X_trainval.shape}")
print(f"Test Size: {X_test.shape}")

# %% [markdown]
# ## 📈 Model 1: Linear Regression

# %%
lr_model = LinearRegression()
lr_scores = cross_val_score(lr_model, X_trainval, y_trainval, cv=cv, scoring='neg_root_mean_squared_error')
print(f"Linear Regression CV RMSE: {-np.mean(lr_scores):.4f}")

# %% [markdown]
# ## 🌲 Model 2: Random Forest (Basic)

# %%
rf_model = RandomForestRegressor(random_state=42)
rf_scores = cross_val_score(rf_model, X_trainval, y_trainval, cv=cv, scoring='neg_root_mean_squared_error')
print(f"Random Forest CV RMSE: {-np.mean(rf_scores):.4f}")

# %% [markdown]
# ## 🔍 Hyperparameter Tuning: Random Forest

# %%
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42),
                           param_grid, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_trainval, y_trainval)

print("Best parameters:", grid_search.best_params_)
print("Best CV RMSE:", -grid_search.best_score_)

# %% [markdown]
# ## ✅ Final Model Evaluation

# %%
final_model = grid_search.best_estimator_
final_model.fit(X_trainval, y_trainval)
y_pred = final_model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")

# %% [markdown]
# ## 📊 Visualization: Actual vs Predicted

# %%
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual PE")
plt.ylabel("Predicted PE")
plt.title("Actual vs Predicted Energy Output")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid()
plt.show()
