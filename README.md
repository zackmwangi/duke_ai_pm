# Power plant model project

This is a simple model development project whose goal is to predict the electrical energy output of a power plant given a dataset.

## Problem definition and model type
This is a supervised learning problem since we have a dataset of labelled features to use.
We are trying to predict a continuous numeric variable(Net hourly electrical energy output, PE), hence this is a regression problem.

## Features
Given the features in the csv data, it is reasonable to assume the following will affect the power output and hence should be considered as predictors:

- Temperature (AT) 
- Ambient Pressure (AP)
- Relative Humidity (RH)
- Exhaust Vacuum (V)

## Algorithm selection
Given the nature of the problem, we will use two different algorithms:
- Linear Regression: A simple baseline model.
- Random Forest: An ensemble method that can capture non-linear relationships.

## Methodology and algorithm Evaluation

- We will split the data into training, validation and test sets.
- we will hold out test until final model evaluation.
- We will use cross-validation to provide a more robust estimate of model performance since we have limited data.
- We will further use K-Fold cross-validation, where we use K=5 for 5-fold cross-validation.

Common output metrics for regression algorithms that we can use to compare and evaluate include:

- Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values. Penalizes larger errors more heavily.

- Root Mean Squared Error (RMSE): The square root of MSE, providing errors in the same units as the target variable.

- Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual values. Less sensitive to outliers than MSE/RMSE.  

- R-squared (R²): Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher R² indicates a better fit.

We will pick the algorithm with a lower RMSE value as the better perfoming alternative.

## Interpretation
 
 We'll train the best performing model (based on CV) on the entire training+validation set and evaluate its performance on the held-out test set.

 We will also plot a visulaization of actual versus predicted values to check if there is a fitness as expected.