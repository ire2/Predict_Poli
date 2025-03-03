import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

train_data = pd.read_csv("data/training_data_prepped.csv")
val_data = pd.read_csv("data/validation_data_prepped.csv")

if 'grass' in train_data.columns:
    target_column = 'grass'
else:
    target_column = [col for col in train_data.columns if 'grass' in col][0]

print(f"Using '{target_column}' as the target column.")

label_encoder = LabelEncoder()
train_data[target_column] = label_encoder.fit_transform(train_data[target_column])
val_data[target_column] = label_encoder.transform(val_data[target_column])

X_train = train_data.drop(columns=[target_column]).values
y_train = train_data[target_column].values
X_val = val_data.drop(columns=[target_column]).values
y_val = val_data[target_column].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:", Counter(y_train_balanced))

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train_balanced, y_train_balanced)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

best_model.save_model("xgboost_model.json")
print("Model saved as xgboost_model.json")
