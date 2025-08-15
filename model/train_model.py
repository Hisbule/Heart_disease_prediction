import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset (replace with your dataset file path)
df = pd.read_csv("heart.csv")  # Change filename if different

# 2. Split features and target
X = df.drop("target", axis=1)   # 'target' column contains labels (0/1)
y = df["target"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)

# 5. Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# 6. Compare & choose best model
if rf_acc > log_acc:
    best_model = rf_model
    best_name = "Random Forest"
    best_acc = rf_acc
else:
    best_model = log_model
    best_name = "Logistic Regression"
    best_acc = log_acc

print(f"Best Model: {best_name} with accuracy: {best_acc:.4f}")
print(classification_report(y_test, best_model.predict(X_test)))

# 7. Save the best model
joblib.dump(best_model, "model/heart_model.joblib")
print("Model saved to model/heart_model.joblib")

