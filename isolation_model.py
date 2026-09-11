import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest

from sklearn.metrics import classification_report

# Load dataset

train = pd.read_csv("data/UNSW_NB15_training-set.csv")

# Remove ID

train = train.drop(columns=["id"])

# Keep features only

X = train.drop(columns=["attack_cat", "label"])

# Convert categorical features

X = pd.get_dummies(

    X,

    columns=["proto", "service", "state"]

)

# Clean data

X = X.replace([float("inf"), float("-inf")], 0)

X = X.fillna(0)

# Use NORMAL traffic for training

normal_data = X[train["label"] == 0]

print("Normal records:", len(normal_data))

print("Features:", normal_data.shape[1])

# Create Isolation Forest

model = IsolationForest(

    n_estimators=200,

    contamination=0.15,

    random_state=42,

    n_jobs=-1

)

# Train

print("\nTraining Isolation Forest...")

model.fit(normal_data)

print("Training completed!")

# Predict entire dataset

predictions = model.predict(X)

# Isolation Forest:

#  1  = normal

# -1  = anomaly

predictions = pd.Series(predictions).map({

    1: 0,

    -1: 1

})

# Actual labels

actual = train["label"]

print("\n==============================")

print("ISOLATION FOREST RESULTS")

print("==============================")

print(

    classification_report(

        actual,

        predictions,

        target_names=["Normal", "Abnormal"],

        zero_division=0

    )

)

# SAVE MODEL
joblib.dump(model, "anomaly_detector.joblib")

print("Isolation Forest model saved!")
