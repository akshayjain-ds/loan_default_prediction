# Loan Default Decision Engine

## Project description and purpose
The purpose of this engine is to predict the likelihood of an incoming laon/credit application to be a defaulter if approved

## Features
1. Input Features: ['checking_balance', 'months_loan_duration', 'credit_history', 'purpose',
 'amount', 'savings_balance', 'employment_length', 'installment_rate',
 'personal_status', 'other_debtors', 'residence_history', 'property',
 'age', 'installment_plan', 'housing', 'existing_credits',
 'dependents', 'foreign_worker', 'job']. 
2. ['gender'] is not used as a feature to avoid any discrimination in our decision making and to make our AI application fair
3. ['telephone'] is kind of an id variable, so ignored from analysis

## Exploratory Data Analysis
EDA is done on all features in oder to:
1. see if there any missing values in a feature and apply imputation and scaling (if applicable and neccesary)
2. see the relationship between input features and target variable

## Transformations and Feature Engineering
1. Transfomed ['residence_history', 'credit_history'] features into months
2. Applied target encodings on all categorical features
3. Kept the remaining numerical features as it is

## Model
1. Lot of missing values in numeric features, but tree based models like xgboost, catboost and lightgbm can handle it pretty well  
2. Trained xgboost/LightGBM/CATBoost models with hyperparameter tuning (using hyperopt) is trained on 80% random training data and tested on remaining 20%
3. XGBoost model came out to be the most stable and accurate
4. The output probabilities from xgb model are calibrated based on 30% default rate
5. AUC: Train is 89%, Test is 86%
6. Shap local level explanation of features is also added for each application.
7. A model definition 'loan_default_engine_definition.py' is created in order to score the engine on incoming json data 
that can be deployed in production for real time scoring

All the relevant artefacts are present in ./artefacts directory

## Decisions
Based on precision-recall analysis on the predicted score, for each application, the decision is divided into 1 of the 3 categories:
1. 'reject': high chance of being a defaulter if application approved, reject the application automatically due to high risk. A rejection reason is also added along with the decision in order to see why the application is risky
2. 'mkyc': medium chance of being a defaulter if application approved, route the application to a loan analyst for further investigation
3. 'approved': low chance of being a defaulter if application approved, can be approved automatically with almost no default risk

## Run
Open the  Pycharm run configuration and make the following changes:

1. Configure the venv interpreter for the project
1. install requirements.txt
1. run main.py
1. file credit_decisions.csv will be created having decisions made on input data ./data/credit.csv