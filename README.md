# AI-Based Network Problem Detection and Offline Assistant

## 📌 Project Overview

This project is an AI-based network monitoring and problem detection system designed to identify abnormal network activity and assist with basic network troubleshooting.

The system uses Machine Learning techniques to detect possible network problems and provides an offline assistant interface for users.

## 🎯 Objectives

- Detect abnormal network behavior using Machine Learning.
- Identify possible network problems from network data.
- Provide an offline troubleshooting assistant.
- Reduce the need for continuous internet connectivity.
- Display results through a simple and user-friendly interface.

## 🛠️ Technologies Used

- Python
- Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Plotly
- Joblib
- SQLite

## 🤖 Machine Learning

The project uses Machine Learning models for network problem detection and classification.

The training data is processed and used to identify patterns associated with abnormal network activity.

## 📂 Project Structure

```text
ai-based-detection/
│
├── app.py
├── isolation_model.py
├── python_train_model.py
├── streamlit_run_app.py
├── anomaly_detector.joblib
├── data/
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
├── requirements.txt
└── README.md
