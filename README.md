# Wine Quality Prediction with Random Forest

A machine learning project that predicts wine quality scores from 
physicochemical properties (acidity, sugar, pH, alcohol, etc.) using 
a Random Forest classifier, and analyzes how training-set size affects 
model accuracy and prediction speed.

## 📊 Dataset
[UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) 
— 6,497 wines (red + white) with 11 numeric features and a quality score (3–9).

## 🎯 What This Project Does
- Preprocesses the data (encodes wine type, imputes missing values)
- Trains a Random Forest classifier on **50%** and **70%** training splits
- Compares accuracy and prediction time across splits
- Visualizes results with bar charts and confusion matrices

## 🔍 Key Findings
| Training Split | Accuracy | Prediction Time |
|----------------|----------|-----------------|
| 50%            | 64.1%    | 0.046 s         |
| 70%            | **68.2%**| 0.036 s         |

- Increasing training data from 50% → 70% improved accuracy by 
  **+4.0 percentage points** (64.1% → 68.2%).
- Prediction time *decreased* at the 70% split — but only because the 
  test set shrinks proportionally (3,248 → 1,949 rows). Per-row 
  prediction cost stays roughly constant, so this is a split-size 
  artifact, not a model-speed improvement.
- Confusion matrices show the model's errors concentrate in **adjacent 
  quality classes** (5↔6, 6↔7), consistent with the ordinal nature of 
  the target variable and the dataset's class imbalance.

## 🛠️ Tech Stack
Python · pandas · scikit-learn · matplotlib · seaborn

## 📈 Outputs
- `results/wine_quality_prediction_results.csv` — accuracy & timing table
- Bar charts comparing accuracy and prediction time across splits
- Confusion matrices for both training splits

## 🔮 Future Improvements
- Treat `quality` as ordinal (e.g., `mord` OrdinalClassifier or regression)
- Address class imbalance with `class_weight='balanced'`
- Hyperparameter tuning via `GridSearchCV`
- Compare against baseline models (Logistic Regression, Gradient Boosting)


  wine-quality-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── wine_quality_analysis.py       ← rename your .py file
├── data/
│   └── winequalityN.csv
└── results/
    └── wine_quality_prediction_results.csv
