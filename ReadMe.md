# Loan Default Decision Engine

## Project description and purpose
The purpose of this engine is to predict the likelihood of an incoming laon/credit application to be a defaulter if approved

## Features
1. Input Features: ['checking_balance', 'months_loan_duration', 'credit_history', 'purpose',
 'amount', 'savings_balance', 'employment_length', 'installment_rate',
 'personal_status', 'other_debtors', 'residence_history', 'property',
 'age', 'installment_plan', 'housing', 'existing_credits',
 'dependents', 'foreign_worker', 'job']. 
2. ['gender'] is not used as a feature to avoid any discrimination in our decisioning process and to keep our AI application fair
3. ['telephone'] is kind of an id variable, so ignored from analysis

## Exploratory Data Analysis
EDA is done on all features in oder to:
1. see if there any missing values in a feature and apply imputation and scaling (if applicable and neccesary)
2. see the relationship between input features and target variable

## Transformations and Feature Engineering
1. Transformed ['residence_history', 'credit_history'] features into number of months
2. Applied target encodings on all categorical features: ['credit_history',
  'purpose',
  'personal_status',
  'other_debtors',
  'property',
  'installment_plan',
  'housing',
  'foreign_worker',
  'job']
3. Kept the remaining numerical features as it is: ['checking_balance',
  'months_loan_duration',
  'amount',
  'savings_balance',
  'employment_length_months',
  'residence_history_months',
  'installment_rate',
  'age',
  'existing_credits',
  'dependents']

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

## Business Decisions
Based on precision-recall and roc analysis of the predicted risk score, for each application, the decision is divided into 1 of the 3 categories:
1. 'reject': high chance of being a defaulter if application is approved, reject the application automatically due to high risk. A rejection reason is also added (in the form of most contributing feature name) along with the decision in order to see why the application is risky
2. 'mkyc': medium chance of being a defaulter if application is approved, route the application to a loan analyst for further investigation
3. 'approved': low chance of being a defaulter if application approved, can be approved automatically with almost no default risk

## Business Impact
If we get 1000 loan applications on a daily basis:
1. ~125 applications will be rejected with a much higher risk of 84% compared to 30% default risk (in case of no ML model)
2. ~355 applications will be auto approved with a much lower risk of 1.5% only compared to 30% default risk (in case of no ML model)
3. remaining 520 (almost half) applications will be going for a manual review
Note, based on risk appetite of the business, the number of applications going to either reject, mkyc, approved is customizable by changing the thresholds

## Run
Open the Pycharm run configuration and make the following changes:

1. Configure the venv interpreter for the project
1. install requirements.txt
1. run main.py
1. file credit_decisions.csv will be created having decisions made on input data ./data/credit.csv