import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter  # Missing import for Counter

# Load preprocessed CSV data
train_data = pd.read_csv("data/training_data_prepped.csv")
val_data = pd.read_csv("data/validation_data_prepped.csv")

# Debugging: Check columns in the dataset
print("Columns in training data:", train_data.columns)

# Identify the target column
if 'grass' in train_data.columns:
    target_column = 'grass'
else:
    target_column = [col for col in train_data.columns if 'grass' in col][0]

print(f"Using '{target_column}' as the target column.")

# Encode target column into numeric labels
label_encoder = LabelEncoder()
train_data[target_column] = label_encoder.fit_transform(train_data[target_column])
val_data[target_column] = label_encoder.transform(val_data[target_column])

# Prepare features and labels
X_train = train_data.drop(columns=[target_column]).values
y_train = train_data[target_column].values
X_val = val_data.drop(columns=[target_column]).values
y_val = val_data[target_column].values

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:", Counter(y_train_balanced))

# Build neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

# Train the model
history = model.fit(
    X_train_balanced, y_train_balanced,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, lr_scheduler],
    verbose=2
)

# Evaluate the model
y_pred = (model.predict(X_val) > 0.5).astype(int)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

# Save the model
model.save("neural_network_model.json")
print("Model saved as neural_network_model.json")
