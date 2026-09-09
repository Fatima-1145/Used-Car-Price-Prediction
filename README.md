# Used Car Price Prediction

This project was built as part of the Nolyth AI Bootcamp Sprint 2.

The goal of this project is to apply classical machine learning techniques to a real-world used-car dataset and build:

1. A regression model to estimate the price of a used car.
2. A classification model to predict the car's price category.
3. A Streamlit application for user-friendly predictions.

---

## Project Objectives

### Regression

The regression task predicts the estimated price of a used car based on:

- Brand
- Mileage
- Fuel type
- Transmission
- Accident history
- Horsepower
- Engine size
- Car age

### Classification

The classification task predicts one of four price categories:

- Budget: up to $30,000
- Mid-range: $30,000–$60,000
- Premium: $60,000–$100,000
- Luxury: above $100,000

The classification target was created from the original price column using fixed price ranges.

---

## Dataset

The dataset contains used-car information including:

- Brand
- Model
- Model year
- Mileage
- Fuel type
- Engine
- Transmission
- Exterior color
- Interior color
- Accident history
- Price

Additional features were created during preprocessing:

- Horsepower
- Engine size
- Car age

---

## Data Cleaning and Preprocessing

The following preprocessing steps were performed:

- Converted price values to numerical format
- Converted mileage values to numerical format
- Handled missing fuel-type values
- Handled missing accident-history values
- Extracted horsepower from the engine description
- Extracted engine size from the engine description
- Created car age from model year
- Removed one suspicious extreme-price record
- Used One-Hot Encoding for categorical features
- Used median imputation for missing numerical values
- Split the dataset into training and testing sets

The final preprocessing pipeline performs numerical imputation using values learned from the training data to reduce preprocessing leakage.

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the relationship between vehicle features and price.

The analysis included:

- Price distribution
- Price outliers
- Mileage vs price
- Car age vs price
- Horsepower vs price
- Correlation analysis
- Brand vs price
- Accident-history analysis
- Fuel-type analysis

### Key Observations

- Higher mileage generally reduces vehicle price.
- Higher horsepower is generally associated with higher prices.
- Newer vehicles tend to have higher prices.
- Luxury brands have significantly higher prices.
- Rare luxury and collector vehicles create large price outliers.

---

# Regression

## Models Tested

The following regression models were tested:

- Dummy Regressor
- Linear Regression
- Decision Tree Regressor
- Controlled Decision Tree Regressor
- Random Forest Regressor

Additional feature-engineering experiments were also performed.

---

## Final Regression Model

The Random Forest Regressor was selected as the final regression model.

### Final Test Metrics

- MAE: approximately $15,419
- RMSE: approximately $82,368
- R²: approximately 0.297

The Random Forest performed better overall than the other tested regression models.

---

## Regression Error Analysis

The model performs better on common used cars than on rare luxury or collector vehicles.

The largest prediction errors were caused by extremely expensive vehicles that had very few similar examples in the dataset.

This is one reason why RMSE is much larger than MAE.

Important features included:

- Mileage
- Engine size
- Horsepower
- Brand
- Car age

---

# Classification

## Price Categories

| Category | Price Range |
|---|---|
| Budget | $0–$30,000 |
| Mid-range | $30,000–$60,000 |
| Premium | $60,000–$100,000 |
| Luxury | Above $100,000 |

---

## Models Tested

The following classification models were tested:

- Dummy Classifier
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Balanced Random Forest Classifier

---

## Classification Results

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| Dummy Classifier | 0.489 | 0.321 | 0.164 |
| Logistic Regression | 0.802 | 0.798 | 0.715 |
| Decision Tree | 0.768 | 0.768 | 0.699 |
| Random Forest | 0.824 | 0.819 | 0.748 |
| Balanced Random Forest | 0.810 | 0.810 | 0.756 |

The Random Forest Classifier was selected as the final classification model because it achieved the best overall accuracy and weighted F1 score.

The Balanced Random Forest slightly improved Macro F1 but reduced overall accuracy.

---

## Classification Challenges

The dataset is imbalanced.

Most vehicles belong to the Budget and Mid-range categories, while Premium and Luxury vehicles are less common.

The model sometimes confuses neighboring price categories, especially:

- Budget vs Mid-range
- Mid-range vs Premium

This can happen because cars close to the category boundaries may have very similar characteristics.

---

# Streamlit Application

A Streamlit application was created to provide a simple interface for making predictions.

The user enters:

- Brand
- Mileage
- Fuel type
- Transmission
- Accident history
- Horsepower
- Engine size
- Car age

The application returns:

- Estimated vehicle price
- Predicted price category

The Streamlit application loads the saved machine learning pipelines directly from `.pkl` files.

---

## Project Structure

```text
Used-Car-Price-Prediction/
│
├── Data/
│   └── used_cars-selected-columns.csv
│
├── used_car_regression.ipynb
├── used_car_classification.ipynb
│
├── used_car_regressor.pkl
├── used_car_classifier.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Fatima-1145/Used-Car-Price-Prediction.git
```

Move into the project folder:

```bash
cd Used-Car-Price-Prediction
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

Run:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## Limitations

The project has several limitations:

- Rare luxury and collector vehicles are difficult to predict because very few similar examples exist in the dataset.
- Extreme price outliers significantly increase RMSE.
- The exact vehicle model was not used in the final model because it had very high cardinality and provided little improvement during experimentation.
- Vehicle condition, location, service history, trim, optional features, and current market conditions may affect real-world prices.
- The Streamlit prediction should be treated as an estimate rather than an exact market valuation.

---

## Final Conclusion

This project demonstrates an end-to-end classical machine learning workflow including:

- Data cleaning
- Feature engineering
- Exploratory Data Analysis
- Regression
- Classification
- Model comparison
- Model evaluation
- Error analysis
- Class imbalance experimentation
- Pipeline-based preprocessing
- Model serialization
- Streamlit integration

The project also shows that model performance should not be judged using only one metric. Error analysis and understanding dataset limitations are also important when evaluating a machine learning model.