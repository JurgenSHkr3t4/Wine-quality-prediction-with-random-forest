import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# --- 1. Load and Preprocess Data ---

try:
    # Load the dataset from the user-provided file
    df = pd.read_csv('winequalityN.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'winequalityN.csv' not found. Please make sure the file is in the correct directory.")
    exit()

# Handle categorical 'type' column
# We convert 'white' and 'red' into numerical values (0 and 1)
label_encoder = LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])

# Handle missing values by filling them with the mean of their respective columns
# This is a common and simple approach for numerical data.
for col in df.columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

print("\nData Preprocessing Complete:")
print(f"Shape of the dataset: {df.shape}")
print(f"Missing values after handling: \n{df.isnull().sum()}")


# Define features (X) and target (y)
# 'quality' is the variable we want to predict.
X = df.drop('quality', axis=1)
y = df['quality']


# --- 2. Function to Train, Evaluate, and Time the Model ---

def train_and_evaluate(X, y, train_size, random_state=42):
    """
    Splits data, trains a RandomForestClassifier, evaluates its performance,
    and measures the time taken.

    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target data.
        train_size (float): The proportion of the dataset to include in the train split.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: A tuple containing accuracy, prediction time, trained model,
               and test data (X_test, y_test).
    """
    print(f"\n--- Starting Model Training with {int(train_size*100)}% Training Data ---")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )

    # Initialize the Random Forest Classifier
    # n_estimators=100 is a good starting point.
    # random_state ensures we get the same results every time we run the code.
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train)

    # Time the prediction process
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    prediction_time = end_time - start_time

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model training and prediction finished.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Prediction Time: {prediction_time:.4f} seconds")

    return accuracy, prediction_time, model, (X_test, y_test, y_pred)

# --- 3. Run Experiments ---

# Experiment 1: 50% training data
acc_50, time_50, model_50, test_data_50 = train_and_evaluate(X, y, train_size=0.5)
X_test_50, y_test_50, y_pred_50 = test_data_50

# Experiment 2: 70% training data
acc_70, time_70, model_70, test_data_70 = train_and_evaluate(X, y, train_size=0.7)
X_test_70, y_test_70, y_pred_70 = test_data_70


# --- 4. Save Results to CSV ---

results_data = {
    'Training Split': ['50%', '70%'],
    'Accuracy': [acc_50, acc_70],
    'Prediction Time (s)': [time_50, time_70]
}
results_df = pd.DataFrame(results_data)

# Save the results to a new CSV file
output_filename = 'wine_quality_prediction_results.csv'
results_df.to_csv(output_filename, index=False)
print(f"\nResults have been saved to '{output_filename}'")
print(results_df)


# --- 5. Visualize the Results ---

print("\nGenerating visualizations...")

# Plot 1: Performance Comparison (Accuracy and Time)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Performance Comparison', fontsize=16)

# Bar plot for Accuracy - UPDATED to address warning
sns.barplot(x='Training Split', y='Accuracy', hue='Training Split', data=results_df, ax=ax1, palette='viridis', legend=False)
ax1.set_title('Prediction Accuracy vs. Training Size')
ax1.set_ylabel('Accuracy Score')
ax1.set_ylim(0, 1) # Accuracy is between 0 and 1
for index, value in enumerate(results_df['Accuracy']):
    ax1.text(index, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontsize=12)


# Bar plot for Prediction Time - UPDATED to address warning
sns.barplot(x='Training Split', y='Prediction Time (s)', hue='Training Split', data=results_df, ax=ax2, palette='plasma', legend=False)
ax2.set_title('Prediction Time vs. Training Size')
ax2.set_ylabel('Time (seconds)')
for index, value in enumerate(results_df['Prediction Time (s)']):
    ax2.text(index, value + 0.001, f'{value:.3f}', ha='center', va='bottom', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Plot 2: Confusion Matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Confusion Matrix Comparison', fontsize=16)

# Confusion Matrix for 50% split
cm50 = confusion_matrix(y_test_50, y_pred_50)
sns.heatmap(cm50, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
ax1.set_title('Confusion Matrix (50% Training Data)')
ax1.set_xlabel('Predicted Quality')
ax1.set_ylabel('True Quality')

# Confusion Matrix for 70% split
cm70 = confusion_matrix(y_test_70, y_pred_70)
sns.heatmap(cm70, annot=True, fmt='d', cmap='Greens', ax=ax2,
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
ax2.set_title('Confusion Matrix (70% Training Data)')
ax2.set_xlabel('Predicted Quality')
ax2.set_ylabel('True Quality')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\nAnalysis complete.")
