# Inflation Prediction Using Multiple Linear Regression

## Project Overview
This project aims to predict annual inflation rates using historical economic indicators. Multiple linear regression is applied to forecast the inflation rate based on the following key features: 
- **Annual GDP Growth Rates (%)**
- **Average Annual Exchange Rates (KSHS vs USD)**
- **Annual CPI Rates (%)**
- **Annual Interest (Lending) Rates (%)**

The target variable is:
- **Annual Inflation Rates (%)**

## Dataset
The dataset consists of economic data collected over several years, and the columns are structured as follows:

| Column Name                                  | Description                                                    |
|----------------------------------------------|----------------------------------------------------------------|
| `YEARS`                                      | The year in which the data was recorded                        |
| `ANNUAL GDP GROWTH RATES (%)`                | The percentage growth of GDP in that year                      |
| `AVERAGE ANNUAL EXCHANGE RATES (KSHS vs USD)`| The average exchange rate between KSHS and USD for that year    |
| `ANNUAL CPI RATES (%)`                       | The Consumer Price Index rates for the year                    |
| `ANNUAL INTEREST (LENDING) RATES (%)`        | The average annual lending interest rates                      |
| `ANNUAL INFLATION RATES (%)`                 | The percentage inflation rate for that year (Target Variable)  |

## Project Structure

- **data/**: This folder contains the dataset file (CSV or Excel format).
- **src/**: Contains the Python code used for data processing and model training.
  - `preprocessing.py`: Script to clean and preprocess the dataset.
  - `train_model.py`: Script to train the multiple linear regression model.
  - `evaluate_model.py`: Script to evaluate the model's performance.
- **notebooks/**: Jupyter notebooks for exploratory data analysis (EDA) and model experimentation.
- **models/**: Saved models after training.

## Getting Started

### Prerequisites

Make sure you have the following installed:
- Python 3.7+
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
Dataset Preprocessing
Missing Data: Any missing values in the dataset are handled using mean imputation or removing rows/columns with excessive missing data.
Feature Scaling: Features such as GDP Growth, CPI Rates, and Exchange Rates are scaled using standardization to improve model performance.
Feature Selection: The features YEARS, GDP Growth, Exchange Rates, CPI Rates, and Interest Rates are used as predictors, while the Inflation Rate is the target variable.
Training the Model
The model is trained using the train_model.py script. It performs the following steps:

Train-Test Split: The data is split into training and testing sets (e.g., 80% training, 20% testing).
Model Training: A multiple linear regression model is trained using the selected features.
Model Evaluation: The model is evaluated using metrics like Mean Squared Error (MSE), R² score, and residual analysis.
Run the following command to train the model:

bash
Copy code
python src/train_model.py
Model Evaluation
The performance of the model is evaluated by comparing the predicted inflation rates against the actual values in the test set. The evaluation metrics used include:

Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R-squared (R²)
You can run the evaluation script using:

bash
Copy code
python src/evaluate_model.py
Results
The model's predictions can be plotted to visualize how well it fits the data. Typical outputs include:

Residual Plots: To check for patterns in prediction errors.
Actual vs Predicted Plot: To visualize the performance of the model.
Conclusion
This project demonstrates how multiple linear regression can be applied to predict inflation rates based on various economic factors. By leveraging historical data, the model can provide insights into future inflation trends.

Next Steps
Improve feature selection or engineering to enhance prediction accuracy.
Experiment with other models like Ridge or Lasso regression for better regularization.
Add more features like unemployment rates or trade balance to improve predictions.
