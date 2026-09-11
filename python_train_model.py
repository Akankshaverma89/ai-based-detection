import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
train = pd.read_csv("data/UNSW_NB15_training-set.csv")


# 2. Remove ID
train = train.drop(columns=["id"])


# 3. Separate features and target
X = train.drop(columns=["attack_cat", "label"])
y = train["attack_cat"]


# 4. Convert categorical features to numbers
X = pd.get_dummies(
    X,
    columns=["proto", "service", "state"]
)


# 5. Clean data
X = X.replace([float("inf"), float("-inf")], 0)
X = X.fillna(0)


# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 7. Create Random Forest
model = RandomForestClassifier(
    n_estimators=250,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# 8. Train
print("\nTraining improved Random Forest...")

model.fit(X_train, y_train)
print("Training completed!")


joblib.dump(model, "problem_classifier.joblib")

print("Random Forest model saved!")


# 9. SAVE TRAINED MODEL
joblib.dump(model, "model.pkl")

print("Model saved successfully as model.pkl")


# 10. Predict
y_pred = model.predict(X_test)


# 11. Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("IMPROVED MODEL RESULTS")
print("==============================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)