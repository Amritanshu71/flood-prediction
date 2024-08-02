import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv("C:/Users/amrit/OneDrive/Desktop/Flood_Prediction/flood.csv")
print(df.head())
print(df.info())
print(df.describe())

# Data Preprocessing
features = df.drop(columns=['FloodProbability'])
target = df['FloodProbability']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

sns.pairplot(df[['MonsoonIntensity', 'TopographyDrainage', 'Deforestation', 'Urbanization', 'ClimateChange', 'FloodProbability']])
plt.show()

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_train_lin = lin_reg.predict(X_train)
y_pred_test_lin = lin_reg.predict(X_test)
train_rmse_lin = mean_squared_error(y_train, y_pred_train_lin, squared=False)
test_rmse_lin = mean_squared_error(y_test, y_pred_test_lin, squared=False)
print(f'Train RMSE (Linear Regression): {train_rmse_lin}')
print(f'Test RMSE (Linear Regression): {test_rmse_lin}')

# Decision Tree Regressor
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train, y_train)
y_pred_train_dt = dt_reg.predict(X_train)
y_pred_test_dt = dt_reg.predict(X_test)
train_rmse_dt = mean_squared_error(y_train, y_pred_train_dt, squared=False)
test_rmse_dt = mean_squared_error(y_test, y_pred_test_dt, squared=False)
print(f'Train RMSE (Decision Tree): {train_rmse_dt}')
print(f'Test RMSE (Decision Tree): {test_rmse_dt}')

# Random Forest Regressor
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_train_rf = rf_reg.predict(X_train)
y_pred_test_rf = rf_reg.predict(X_test)
train_rmse_rf = mean_squared_error(y_train, y_pred_train_rf, squared=False)
test_rmse_rf = mean_squared_error(y_test, y_pred_test_rf, squared=False)
print(f'Train RMSE (Random Forest): {train_rmse_rf}')
print(f'Test RMSE (Random Forest): {test_rmse_rf}')

# Randomized Search for Random Forest Regressor
param_dist_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search_rf = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist_rf,
    n_iter=50,  # Number of parameter settings that are sampled
    cv=3,  # Number of folds in cross-validation
    verbose=1,  # Controls the verbosity: the higher, the more messages
    n_jobs=-1,  # Use all available cores
    random_state=42
)

random_search_rf.fit(X_train, y_train)
print(random_search_rf.best_params_)
best_model_rf = random_search_rf.best_estimator_
y_pred_test_best_rf = best_model_rf.predict(X_test)
test_rmse_best_rf = mean_squared_error(y_test, y_pred_test_best_rf, squared=False)
print(f'Test RMSE (Best Random Forest): {test_rmse_best_rf}')

# Feature Importance
importances = best_model_rf.feature_importances_
feature_names = features.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()

# Save the best model and scaler
joblib.dump(best_model_rf, 'best_model_rf.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Define the subset of features you want to use
selected_features = ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation', 'Urbanization', 'ClimateChange']
subset_features = df[selected_features]

# Fit a new scaler on these features
new_scaler = StandardScaler()
new_scaler.fit(subset_features)

# Save this new scaler
joblib.dump(new_scaler, 'subset_scaler.pkl')
