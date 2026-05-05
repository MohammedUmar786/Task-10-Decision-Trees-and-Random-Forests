# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("WineQT.csv")

# Drop unnecessary columns if present
if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)

print("Dataset Shape:", df.shape)
print(df.head())

# ==============================
# 3. Preprocessing
# ==============================
# Features and Target
X = df.drop('quality', axis=1)
y = df['quality']

# Convert to classification (optional: binary classification)
# Example: Good (>=6) vs Bad (<6)
y = y.apply(lambda x: 1 if x >= 6 else 0)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 4. Decision Tree Model
# ==============================
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluation
print("\n=== Decision Tree Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d')
plt.title("Decision Tree Confusion Matrix")
plt.show()

# ==============================
# 5. Visualize Decision Tree
# ==============================
plt.figure(figsize=(15, 8))
plot_tree(dt_model, feature_names=X.columns, class_names=["Bad", "Good"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# ==============================
# 6. Random Forest Model
# ==============================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
print("\n=== Random Forest Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d')
plt.title("Random Forest Confusion Matrix")
plt.show()

# ==============================
# 7. Feature Importance
# ==============================
importances = rf_model.feature_importances_
features = X.columns

feature_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:\n", feature_df)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title("Feature Importance (Random Forest)")
plt.show()

# ==============================
# 8. Model Comparison
# ==============================
print("\n=== Model Comparison ===")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")# Task-10-Decision-Trees-and-Random-Forests
