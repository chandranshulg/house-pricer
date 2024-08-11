import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import numpy as np
import joblib

# 1. Load the dataset
data = {
    'Size (sqft)': [1500, 2000, 2500, 1800, 2200, 2300, 2600, 1700, 2100, 1900,
                    1600, 2700, 3000, 2800, 2400, 3200, 3100, 3300, 3500, 1400],
    'Bedrooms': [3, 4, 4, 3, 4, 3, 5, 3, 4, 3, 3, 5, 4, 4, 3, 5, 5, 4, 6, 2],
    'Age': [10, 15, 5, 20, 7, 12, 8, 30, 18, 14, 16, 5, 3, 6, 10, 7, 2, 1, 4, 25],
    'Price': [300000, 450000, 500000, 380000, 480000, 490000, 510000, 350000, 420000, 390000,
              320000, 530000, 550000, 600000, 470000, 620000, 610000, 640000, 700000, 310000]
}

# 2. Create a DataFrame
df = pd.DataFrame(data)

# 3. Split the data into features (X) and target (y)
X = df[['Size (sqft)', 'Bedrooms', 'Age']]
y = df['Price']

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create a pipeline with StandardScaler and RandomForestRegressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('model', RandomForestRegressor(random_state=42))  # Model
])

# 6. Set up hyperparameter grid for GridSearchCV
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# 7. Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# 8. Train the model with hyperparameter tuning
grid_search.fit(X_train, y_train)

# 9. Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# 10. Evaluate the model on the test set
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("R^2 Score:", r2)

# 11. Save the model for future use
joblib.dump(best_model, 'house_price_model.pkl')

# 12. Load the model and make a new prediction (example)
loaded_model = joblib.load('house_price_model.pkl')
new_data = np.array([[2200, 4, 15]])  # Example: Size=2200 sqft, Bedrooms=4, Age=15
new_prediction = loaded_model.predict(new_data)
print(f"Predicted price for the new data: ${new_prediction[0]:,.2f}")

# 13. Print feature importance
feature_importance = best_model.named_steps['model'].feature_importances_
features = X.columns

print("\nFeature Importances:")
for feature, importance in zip(features, feature_importance):
    print(f"{feature}: {importance:.4f}")
