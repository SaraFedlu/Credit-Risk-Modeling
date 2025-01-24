# BatiBank Credit Scoring Model

## Overview
This project involves developing a comprehensive credit scoring system for Bati Bank's "Buy-Now-Pay-Later" service. The system aims to classify users into high-risk and low-risk categories, assign credit scores, and predict optimal loan amounts and durations using a data-driven approach.

## Features
- **Data Analysis:** Exploratory Data Analysis (EDA) to understand data structure, identify patterns, and detect outliers.
- **Feature Engineering:** Create new aggregate and extracted features, encode categorical variables, handle missing values, and normalize numerical features.
- **Default Estimation:** Classify users into good and bad credit groups using RFMS formalism and Weight of Evidence (WoE) binning.
- **Predictive Modeling:** Develop machine learning models to estimate default probability, assign credit scores, and optimize loan recommendations.
- **Model Serving:** Build a REST API for real-time credit scoring predictions.

## Data Description
The dataset includes the following fields:
- **TransactionId:** Unique identifier for each transaction.
- **BatchId:** Identifier for transaction batches.
- **AccountId, SubscriptionId, CustomerId:** Identifiers for customers and their subscriptions.
- **CurrencyCode, CountryCode:** Currency and geographical information.
- **ProviderId, ProductId, ProductCategory:** Details of items bought and their categories.
- **ChannelId:** Customer access channels (e.g., web, Android, iOS).
- **Amount, Value:** Transaction amounts and their absolute values.
- **TransactionStartTime:** Timestamp of transaction initiation.
- **PricingStrategy:** Merchant pricing structure category.
- **FraudResult:** Fraud detection status (1 = Yes, 0 = No).

## Tasks
### Task 1: Understanding Credit Risk
- Define default and its significance in credit scoring.
- Understand the Basel II Capital Accord guidelines for credit risk management.

### Task 2: Exploratory Data Analysis (EDA)
1. **Data Structure:** Inspect rows, columns, and data types.
2. **Summary Statistics:** Analyze central tendencies, dispersion, and distributions.
3. **Numerical Features:** Visualize distributions and detect skewness/outliers.
4. **Categorical Features:** Analyze frequency and variability of categories.
5. **Correlation Analysis:** Identify relationships between numerical features.
6. **Missing Values:** Detect and handle missing data.
7. **Outlier Detection:** Use box plots to identify anomalies.

### Task 3: Feature Engineering
1. **Aggregate Features:**
   - Total Transaction Amount
   - Average Transaction Amount
   - Transaction Count
   - Standard Deviation of Transaction Amounts
2. **Extract Features:**
   - Transaction Hour, Day, Month, Year
3. **Encode Categorical Variables:**
   - One-Hot Encoding
   - Label Encoding
4. **Handle Missing Values:**
   - Imputation (e.g., mean, median, KNN)
   - Removal (if minimal)
5. **Normalize/Standardize Features:**
   - Normalization ([0, 1] range)
   - Standardization (mean = 0, SD = 1)

### Task 4: Modeling
1. **Model Development:**
   - Logistic Regression
   - Decision Trees
   - Random Forest
   - Gradient Boosting Machines (GBM)
2. **Model Training:**
   - Split data into training and testing sets.
   - Perform hyperparameter tuning using Grid Search or Random Search.
3. **Evaluation Metrics:**
   - Accuracy, Precision, Recall, F1 Score, ROC-AUC.

### Task 5: Model Serving API
1. **Framework:** Choose Flask, FastAPI, or Django REST framework.
2. **Endpoints:** Define endpoints to accept input data and return predictions.
3. **Deployment:** Deploy API on a web server or cloud platform.

## Getting Started
### Prerequisites
- Python 3.7+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, Flask/FastAPI, xverse, woe.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SaraFedlu/Credit-Risk-Modeling.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Credit-Risk-Modeling
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Run the EDA script to understand the data:
   ```bash
   python eda.py
   ```
2. Perform feature engineering:
   ```bash
   python feature_engineering.py
   ```
3. Train and evaluate models:
   ```bash
   python train_models.py
   ```
4. Start the API server:
   ```bash
   python app.py
   ```

### API Endpoints
- **POST /predict:** Accepts transaction data and returns risk predictions and credit scores.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributors
- **Sara Fedlu**: Analytics Engineer