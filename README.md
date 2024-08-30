# House Price Prediction Model

This project involves building a machine learning model to predict house prices based on features like size, number of bedrooms, and age of the house. The model is developed using a RandomForestRegressor, and hyperparameter tuning is performed using GridSearchCV.

## Overview

The House Price Prediction model uses historical data to train a RandomForestRegressor model. The model is then evaluated based on metrics such as Root Mean Squared Error (RMSE) and R^2 Score. The trained model is saved for future predictions.

## Features

- **Data Preprocessing:** The data is split into training and testing sets, and features are scaled using StandardScaler.
- **Model Training:** A RandomForestRegressor model is trained, with hyperparameters tuned using GridSearchCV.
- **Model Evaluation:** The model is evaluated on the test set using RMSE and R^2 Score.
- **Model Persistence:** The trained model is saved as a `.pkl` file for future use.
- **Feature Importance:** The model's feature importances are printed to understand the contribution of each feature.

## Technologies Used

- **Python:** The programming language used for the project.
- **Pandas:** For data manipulation and analysis.
- **Scikit-learn:** For model building, hyperparameter tuning, and evaluation.
- **Joblib:** For saving and loading the trained model.

## Future Enhancements

- **Expand Dataset:** Include more features and data points to improve model accuracy.
- **Model Comparison:** Evaluate and compare the performance of different regression models.
- **Web Interface:** Create a simple web interface to allow users to input house features and get price predictions.

## License

This project is open-source and available under the MIT License. Feel free to use, modify, and distribute as needed.

## Author

Created by Chandranshu.
