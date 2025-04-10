# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 2. Load Dataset
df = pd.read_csv("C:/Users/HP/overstimulation_dataset.csv")
print("Shape:", df.shape)
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# 3. Descriptive Statistics
print("\nDescriptive Stats:\n", df.describe())

# 4. Overstimulation Class Distribution
overstim_counts = df['Overstimulated'].value_counts(normalize=True) * 100
print("\nOverstimulation Rate:\n", overstim_counts)

# 5. Visualize Class Distribution
sns.countplot(x='Overstimulated', data=df)
plt.title("Overstimulation Distribution")
plt.xlabel("Overstimulated (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 7. Compare Key Features by Overstimulation
key_features = ['Screen_Time', 'Sleep_Hours', 'Stress_Level', 'Anxiety_Score', 'Exercise_Hours']
for col in key_features:
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=df, x=col, hue='Overstimulated', fill=True)
    plt.title(f"{col} Distribution by Overstimulation")
    plt.show()

# 8. Quantitative Comparisons
print("\nMean Screen Time:")
print(df.groupby('Overstimulated')['Screen_Time'].mean())

print("\nMean Sleep Hours:")
print(df.groupby('Overstimulated')['Sleep_Hours'].mean())

print("\nAnxiety & Depression Scores:")
print(df.groupby('Overstimulated')[['Anxiety_Score', 'Depression_Score']].mean())

print("\nExercise Hours:")
print(df.groupby('Overstimulated')['Exercise_Hours'].mean())

# 9. Prepare Data for Modeling
X = df.drop(columns=['Overstimulated'])
y = df['Overstimulated']

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 10. Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\nLogistic Regression Results:\n", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# 11. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("\nDecision Tree Results:\n", classification_report(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# 12. Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Results:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# 13. Feature Importance (Random Forest)
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.nlargest(10)
top_features.plot(kind='barh')
plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Importance Score")
plt.show()
